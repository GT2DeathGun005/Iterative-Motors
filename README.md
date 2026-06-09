# AIcar - Hybrid BC-RL Architecture (IBM AI RACING LEAGUE 2026)

Agente autonomo per TORCS sviluppato per partecipare alla **IBM AI RACING LEAGUE 2026**. Il modello viene addestrato in due fasi: imita giri umani puliti con Behavioral Cloning, poi viene raffinato online con TD3+BC mantenendo un ancoraggio costante ai dati esperti.

La pipeline attuale usa una sola architettura viva:

- stato base 29D, stacked in 87D (`t-12`, `t-6`, `t`);
- Actor continuo con backbone 4x512 + `continuous_head` per `steer`, `accel`, `brake`;
- cambio marcia deterministico in `gearing.py`;
- ambiente TORCS con azione unica `[steer, accel, brake, gear]`;
- reward da corsa minimalista: velocità in avanti, validità pista, anti-zigzag leggero;
- replay online, expert buffer permanente ed elite buffer per self-imitation.

Il wrapper espone solo il contratto usato dalla pipeline: la policy controlla gas/freno, `gearing.py` controlla la marcia, TORCS gira senza input visivo della rete.

### Paper di riferimento

Le scelte correnti sono ancorate a tre linee di letteratura:

- **Fujimoto & Gu, 2021 - TD3+BC**: base della loss Actor `-lambda Q + (pi-a)^2`, normalizzazione del termine RL con `lambda / mean(|Q|)` e normalizzazione degli stati.
- **Bojarski et al., 2016 - End to End Learning for Self-Driving Cars**: ispirazione per l'augmentation di recupero, dove stati perturbati insegnano alla rete a rientrare in traiettoria.
- **Beeson & Montana, 2022 - Conservative/Relaxed Policy Constraint per offline-to-online RL**: riferimento per la fase separata di refinement, in cui il vincolo imitativo viene allentato solo quando la policy è stabile in plateau.

---

## Perché Funziona

Il Behavioral Cloning da solo guida bene solo finché l'auto resta vicino alla distribuzione umana. Appena l'agente sbaglia traiettoria, incontra stati che il dataset non contiene e accumula errore. Il TD3+BC risolve il problema senza distruggere ciò che il BC ha imparato:

1. **Warm-start dal BC**: l'Actor parte già capace di sterzare, accelerare e frenare in modo plausibile.
2. **Ancora BC costante**: nei batch expert l'Actor paga una penalità se si allontana dall'azione umana. Questo impedisce drift improvvisi.
3. **Masking rigoroso**: la penalità imitativa si applica solo ai campioni marcati expert. Sugli stati online sporchi l'Actor può seguire il Critic e imparare recuperi.
4. **Expert buffer separato**: i giri umani non vengono persi dalla FIFO del replay online. Ogni batch mantiene un riferimento umano.
5. **Cambio deterministico**: la marcia non è predetta dalla rete. `gearing.compute_gear()` usa velocità, gas applicato, rpm e cooldown, eliminando oscillazioni di marcia e mismatch tra training/test.
6. **Reward minimale**: il reward incentiva il progresso e punisce uscita pista, spin, stallo e giro non completato. L'agente resta libero di scegliere velocità e staccate.
7. **Normalizzazione coerente**: scaling fisico fisso + normalizzazione mean/std condivisa da BC, TD3 e test. La rete vede lo stesso spazio in ogni fase.

---

## Pipeline

```
data_collection.py  ->  behavioral_cloning.py  ->  td3_bc.py  ->  test_agent.py
      HDF5                    bc_policy.pth          TD3+BC        eval deterministica
```

| Fase | Script | Output principale |
|---|---|---|
| Raccolta dati | `data_collection.py` | `train_set/laps/lap_*.h5` |
| Behavioral Cloning | `behavioral_cloning.py` | `train_set/checkpoints/bc_policy.pth`, `state_norm.npz` |
| TD3+BC | `td3_bc.py` / `train_rl.sh` | `td3_policy.pth`, `td3_checkpoint.pth`, record `td3_*` |
| Test | `test_agent.py` | telemetria CSV e tempi giro |

---

## Repository

```
AIcar/
├── data_collection.py
├── behavioral_cloning.py
├── td3_bc.py
├── test_agent.py
├── gearing.py
├── train_bc.sh
├── train_rl.sh
├── stop_training.sh
├── gym_torcs/
│   ├── gym_torcs.py
│   ├── snakeoil3_gym.py
│   └── autostart.sh
├── telemetry/
└── train_set/
    ├── laps/
    ├── checkpoints/
    │   ├── backups/
    │   └── buffers/
    └── session_logs/
```

---

## 1. Raccolta Dati

Registra giri umani validi. Il giro viene salvato solo se completato senza superare `|trackPos| > 1.25`.

```bash
python data_collection.py --output_dir train_set --device controller
python data_collection.py --output_dir train_set --device keyboard
```

Raccolta mirata per curve difficili:

```bash
python data_collection.py --output_dir train_set --device controller --segment_only
python data_collection.py --output_dir train_set --device controller --segment_only --zones "670:810,940:1070"
```

I file HDF5 contengono:

- `states`: stato base 29D;
- `actions`: `[steer, accel, brake, gear]`;
- `dist_from_start`: metadato di posizione per analisi e segmentazione.

`dist_from_start` non entra nella rete. Il modello resta 29D perché la posizione assoluta produce mismatch tra train/test e non generalizza alle correzioni locali.

---

## 2. Behavioral Cloning

```bash
./train_bc.sh
```

Oppure:

```bash
python behavioral_cloning.py \
    --dataset train_set/laps \
    --epochs 300 \
    --batch_size 256 \
    --lr 3e-4 \
    --output train_set/checkpoints/bc_policy.pth
```

Il BC carica solo giri interi `lap_[0-9]*.h5`. I segmenti `lap_seg_*.h5` sono esclusi dal BC perché concentrano una sola curva e spostano la media delle azioni su stati sensorialmente simili. Quei segmenti vengono invece usati dal TD3+BC nell'expert buffer, dove il Critic può valutarli senza trasformarli in un target globale cieco alla posizione.

Il training BC usa augmentation di recupero sul 50% del batch:

- perturbazione laterale `trackPos`;
- perturbazione angolare `angle`;
- aggiornamento geometrico dei sensori `track`;
- correzione proporzionale dello sterzo target;
- riduzione del gas e aumento del freno per stati troppo aggressivi.

Il 50% non perturbato preserva la guida pulita sulla traiettoria ideale.

---

## 3. TD3+BC

```bash
./train_rl.sh
TD3_EPISODES=500 ./train_rl.sh
./train_rl.sh --clean
./train_rl.sh --rollback
./train_rl.sh --rollback --actor-freeze-episodes 100 --no-auto-refine
./train_rl.sh --no-auto-refine
./train_rl.sh --refine
```

Lancio diretto:

```bash
python td3_bc.py \
    --bc_weights train_set/checkpoints/bc_policy.pth \
    --episodes 1000 \
    --max_steps 5000 \
    --seed 42
```

Durante il training:

- l'Actor emette azioni TD3 in `[-1, 1]`;
- `accel` e `brake` vengono mappati in `[0, 1]`;
- `accel_final = accel * (1 - brake)` evita pressione simultanea dei pedali senza discontinuità rigide;
- `gearing.compute_gear()` decide la marcia usando velocità, gas applicato, rpm e cooldown;
- il Critic viene aggiornato a ogni step;
- l'Actor viene aggiornato ogni 2 update del Critic dopo 15.000 step di warm-up.

Sampling del batch:

- 25% expert umano;
- 15% elite/self-imitation, se disponibile;
- 60% online.

Se online o elite sono scarsi, il batch viene riempito dall'expert buffer.

---

## 4. Test Deterministico

```bash
SHOW_GUI=1 python test_agent.py
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/td3_det_best_lap.pth --laps 1
SHOW_GUI=1 python test_agent.py --weights train_set/checkpoints/td3_det_best_dist.pth --laps 1
```

Priorità auto-detect:

1. `td3_det_best_lap.pth`
2. `td3_det_best_dist.pth`
3. `td3_det_best_dist_run.pth`
4. `td3_expl_best_lap.pth`
5. `td3_expl_best_dist.pth`
6. `td3_policy.pth`
7. `bc_policy.pth`

I pesi TD3 usano `tanh` su tutti i canali continui. I pesi BC usano `tanh` sullo sterzo e `sigmoid` su gas/freno. `test_agent.py` sceglie la conversione dal nome file (`td3_*` o `bc_*`); per nomi custom usa `--kind rl` o `--kind bc`.

---

## Checkpoint

| File | Significato |
|---|---|
| `bc_policy.pth` | Actor supervisionato da dati umani |
| `td3_policy.pth` | ultimo Actor TD3 salvato |
| `td3_checkpoint.pth` | stato completo per resume: reti, ottimizzatori, step, record |
| `td3_det_best_lap.pth` | miglior giro valido deterministico, candidato submission |
| `td3_det_best_dist.pth` | miglior distanza deterministica assoluta, preservata dopo `--clean` |
| `td3_det_best_dist_run.pth` | miglior distanza deterministica del run corrente |
| `td3_expl_best_lap.pth` | giro valido in rollout esplorativo |
| `td3_expl_best_dist.pth` | miglior distanza in rollout esplorativo |
| `buffers/td3_checkpoint_buffer.npz` | replay online |
| `buffers/td3_checkpoint_elite_buffer.npz` | elite buffer |

Il salvataggio è atomico: buffer prima, checkpoint completo dopo. I backup `.bak` e `.prev` stanno in `train_set/checkpoints/backups/`. Al resume, se un buffer è più nuovo del checkpoint scelto, viene ignorato in favore del backup allineato.

---

## Reward

Nel wrapper TORCS il reward per step è:

```text
progress = (speedX / 50.0) * cos(angle)
reward = progress * 1.5
       - 2.0 * max(0, abs(trackPos) - 1.0)^2
       - 0.05 * abs(steer - last_steer)
```

Terminazioni non valide:

- `|trackPos| > 1.25`: giro invalido, con penalità terminale graduata;
- stallo dopo il transitorio iniziale;
- auto girata in senso opposto;
- timeout/fine episodio senza un `lastLapTime` valido.

Il bonus `+50` viene assegnato nel loop TD3 quando TORCS aggiorna `lastLapTime`, cioè quando un giro valido viene completato. Se l'episodio termina senza un giro valido, il loop TD3 applica un malus di giro incompleto.

Questa formulazione ha funzionato perché non dice all'agente come affrontare una curva. Premia solo avanzamento valido e stabilità minima, lasciando al TD3 la libertà di trovare staccate e velocità migliori dei dati medi umani.

---

## Stato e Normalizzazione

Stato base 29D:

| Indice | Feature | Scaling fisico |
|---|---|---|
| 0 | `angle` | radianti |
| 1-19 | `track[19]` | `/200` |
| 20 | `trackPos` | nessuno |
| 21 | `speedX` | `/50` |
| 22 | `speedY` | `/50` |
| 23 | `speedZ` | `/50` |
| 24-27 | `wheelSpinVel[4]` | `/100` |
| 28 | `rpm` | `/10000` |

Poi `state_norm.npz` applica mean/std alle 29 feature prima dello stacking nella rete. Non aggiungere normalizzazioni locali: BC, TD3 e test devono vedere lo stesso spazio.

---

## Cambio Marcia

`gearing.compute_gear(speed_kmh, accel, rpm, current_gear, steps_since_shift)` è l'unica logica di cambio marcia usata in training, eval e test.

Principi:

- downshift basato sulla velocità, non sugli rpm;
- upshift consentito solo se il gas applicato è sufficiente e gli rpm sono alti;
- soglie con isteresi;
- cooldown dopo ogni cambio.

Questo ha risolto le oscillazioni perché in staccata gli rpm possono salire temporaneamente dopo un downshift, mentre la velocità resta monotona. Guardare la velocità per scalare evita il loop scendi-risali-scendi.

---

## Refinement

La refinement è una fase separata, non il training normale. Si attiva manualmente con `--refine` o automaticamente quando le eval deterministiche restano in plateau:

- Critic non aggiornato;
- loss Critic mostrata solo come diagnostica;
- peso BC ridotto a `0.3`;
- uscita in consolidamento se supera stabilmente il plateau;
- ritorno a `bc_weight=1.0` e Critic riattivato.

Serve quando la policy è stabile ma bloccata sotto una curva difficile: si riduce temporaneamente il vincolo imitativo per sfruttare il valore già appreso dal Critic, poi si consolida di nuovo con l'ancora BC completa.

---

## Invarianti del Progetto

- `TorcsEnv.step()` riceve sempre `[steer, accel, brake, gear]`.
- La rete non predice la marcia.
- Il BC addestra solo `steer`, `accel`, `brake`.
- Il BC carica solo giri interi.
- I segmenti mirati entrano solo nel TD3 expert buffer.
- `dist_from_start` è metadato, non feature.
- Il cambio marcia è sempre `gearing.compute_gear()`.
- La mutual exclusion dei pedali è moltiplicativa.
- La policy di test è deterministica.

---

**IBM AI RACING LEAGUE 2026** - Precision Driving through Hybrid BC-RL.
