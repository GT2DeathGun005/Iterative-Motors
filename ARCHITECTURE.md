# AIcar: Hybrid BC-RL Architecture (IBM AI RACING LEAGUE 2026)

Questo documento è la fonte di verità tecnica del progetto per la partecipazione alla **IBM AI RACING LEAGUE 2026**. Descrive solo l'approccio attuale: Behavioral Cloning continuo, TD3+BC deterministico, cambio marcia algoritmico e ambiente TORCS con azione `[steer, accel, brake, gear]`.

## Paper Usati

- **Fujimoto & Gu, 2021 - TD3+BC**: loss Actor con termine RL normalizzato e penalità BC, target policy smoothing, delayed policy update e normalizzazione dello stato.
- **Bojarski et al., 2016 - End to End Learning for Self-Driving Cars**: data augmentation di recupero, usata qui perturbando `trackPos`, `angle` e sensori pista per insegnare correzioni controllate.
- **Beeson & Montana, 2022 - offline-to-online RL con vincolo di policy rilassato**: base concettuale della refinement separata, in cui `bc_weight` scende temporaneamente solo su plateau e il Critic resta fisso.

---

## 1. Contratto dell'Ambiente

`gym_torcs.TorcsEnv` espone una sola modalità operativa.

```python
env = TorcsEnv(early_termination=True)
next_obs, reward, done, info = env.step([steer, accel, brake, gear])
```

Azione:

- `steer` in `[-1, 1]`;
- `accel` in `[0, 1]`;
- `brake` in `[0, 1]`;
- `gear` intero in `[1, 6]`.

In training/test, gas e freno arrivano dalla policy; la marcia arriva da `gearing.compute_gear()`.

TORCS viene avviato senza fuel e senza damage fisico distruttivo:

```text
torcs -nofuel -nodamage
```

Il danno viene comunque letto dalla telemetria e usato come segnale terminale per il reward.

---

## 2. Stato

Lo stato base è 29D:

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

La rete riceve 87D concatenando tre frame:

```text
[state(t-12), state(t-6), state(t)]
```

Il dataset salva anche `dist_from_start`, ma solo come metadato. Non entra nello stato della rete perché la policy deve reagire alla geometria e alla dinamica dell'auto, non memorizzare posizioni assolute del circuito.

`state_norm.npz` contiene media e deviazione standard delle 29 feature. La normalizzazione statistica viene applicata in modo identico in:

- training BC;
- caricamento expert in TD3;
- rollout TD3;
- test deterministico.

---

## 3. Actor

Actor BC/TD3+BC:

```text
input 87D
-> Linear 512 + LayerNorm + ReLU
-> Linear 512 + LayerNorm + ReLU
-> Linear 512 + LayerNorm + ReLU
-> Linear 512 + LayerNorm + ReLU
-> continuous_head 3D
```

Canali continui:

- `steer`;
- `accel`;
- `brake`.

Nel BC:

- `steer = tanh(out[0])`;
- `accel = sigmoid(out[1])`;
- `brake = sigmoid(out[2])`.

Nel TD3:

- `action = tanh(mean)`;
- `accel = (action[1] + 1) / 2`;
- `brake = (action[2] + 1) / 2`.

Il warm-start dal BC dimezza pesi e bias dei canali `accel/brake` prima di caricarli nel TD3:

```text
(tanh(0.5x) + 1) / 2 ~= sigmoid(x)
```

In questo modo l'Actor TD3 iniziale produce praticamente gli stessi pedali del modello BC.

---

## 4. Cambio Marcia

La marcia è calcolata da `gearing.compute_gear()` in training, eval e test.

Input:

- velocità corrente in km/h;
- gas applicato dopo mutual exclusion;
- rpm;
- marcia corrente;
- step trascorsi dall'ultimo cambio.

Regole:

- downshift guidato dalla velocità;
- upshift solo con gas sufficiente e rpm alti;
- isteresi tra soglie di salita e scalata;
- cooldown dopo ogni cambio.

Perché ha funzionato: in staccata un downshift aumenta gli rpm, quindi una regola basata solo sugli rpm tende a cambiare idea subito. La velocità invece scende in modo monotono; usarla per scalare rende il cambio stabile. Il gate sul gas impedisce upshift mentre l'auto sta frenando.

---

## 5. Mutual Exclusion Pedali

La rete può emettere gas e freno entrambi positivi. Prima dell'invio a TORCS viene applicata:

```text
accel_final = accel * (1 - brake)
```

Questa formula è continua e mantiene uno spazio d'azione liscio. Se il freno è alto, il gas cala naturalmente; se il freno è zero, il gas passa invariato.

La stessa regola è usata in:

- loop TD3 esplorativo;
- eval deterministica in `td3_bc.py`;
- test in `test_agent.py`.

---

## 6. Behavioral Cloning

`behavioral_cloning.py` addestra solo `steer`, `accel`, `brake`.

Loss continua:

- MSE pesato;
- freno pesato più dello sterzo/gas;
- boost del freno quando il target umano frena;
- boost dello sterzo in curva.

Il BC carica solo file `lap_[0-9]*.h5`, cioè giri interi. I segmenti `lap_seg_*.h5` sono esclusi dal BC e usati dal TD3 expert buffer.

Motivo: il BC è cieco alla posizione assoluta e ottimizza una media globale. Segmenti concentrati su una curva stretta sbilanciano la media delle azioni e degradano il comportamento su curve sensorialmente simili. Nel TD3, invece, quei segmenti sono utili perché il Critic valuta la transizione e non costringe la policy a imitare quella curva ovunque.

---

## 7. Augmentation BC

Su circa metà batch, lo stato viene perturbato per simulare errori di traiettoria:

- `trackPos` spostato lateralmente;
- `angle` perturbato;
- sensori `track` aggiornati in modo geometrico;
- sterzo target corretto verso la pista;
- gas ridotto;
- freno aumentato nei casi ad alta velocità/curva.

L'altra metà del batch resta pulita. Questo equilibrio è importante: la rete impara sia recupero da stati sporchi sia fedeltà alla guida ideale.

---

## 8. TD3+BC

TD3+BC usa:

- Actor deterministico;
- Twin Critic;
- target policy smoothing;
- delayed policy update ogni 2 update del Critic;
- warm-up Actor di 15.000 step;
- `gamma = 0.99`;
- `reward_scale = 0.02`;
- learning rate `3e-4` per Actor e Critic;
- gradient clipping a `1.0`.

Loss Actor:

```text
actor_loss = alpha * (-Q(s, pi(s))) + bc_weight * bc_penalty
alpha = 2.5 / mean(abs(Q(s, pi(s))))
bc_weight = 1.0 nel training normale
```

La normalizzazione di `alpha` rende il termine RL robusto alla scala dei Q-value. Il peso BC costante mantiene l'Actor vicino alle azioni umane quando il batch contiene expert data.

---

## 9. Replay Buffer

Ci sono tre sorgenti nel minibatch TD3:

- Expert buffer: giri e segmenti umani caricati da `train_set/laps`;
- Online buffer: transizioni raccolte dall'Actor durante il training;
- Elite buffer: traiettorie dell'agente che superano la soglia record.

Quote:

```text
25% expert
15% elite
60% online
```

Se online o elite non hanno abbastanza campioni, il resto viene riempito dall'expert buffer.

Perché ha funzionato:

- i dati umani restano sempre disponibili;
- l'online buffer porta stati sporchi realmente visitati;
- l'elite buffer permette self-imitation dei record dell'agente;
- gli ultimi 50 step di una traiettoria elite conclusa male non vengono marcati expert, così l'Actor non imita la sequenza che ha causato il crash.

---

## 10. Reward

Reward per step:

```text
progress = (speedX / 50.0) * cos(angle)
reward = progress * 1.5
       - 2.0 * max(0, abs(trackPos) - 1.0)^2
       - 0.05 * abs(steer - last_steer)
```

Terminali con `reward = -10`:

- danno nuovo;
- `|trackPos| > 1.25`;
- stallo dopo 500 step;
- auto rivolta all'indietro.

Bonus:

- `+50` quando TORCS aggiorna `lastLapTime`, cioè quando il giro è valido.

Questa forma ha risolto il problema dell'agente troppo vincolato: non penalizza direttamente la velocità in curva e non prescrive una traiettoria. Premia solo avanzamento valido e lascia al TD3 la ricerca delle staccate.

---

## 11. Done Masking

Nel replay:

- `mask = 0.0` solo per crash/fallimento reale;
- `mask = 1.0` per time-limit;
- dati expert sempre `mask = 1.0`.

Il completamento di un giro umano non è un crash. Azzerare il futuro sui dati expert confonderebbe il Critic, perché la transizione terminale di un giro valido e quella di uno schianto avrebbero la stessa semantica Bellman.

---

## 12. Checkpointing

Salvataggi principali:

- `td3_checkpoint.pth`: stato completo;
- `td3_policy.pth`: ultimo Actor;
- `td3_det_best_lap.pth`: miglior giro valido deterministico;
- `td3_det_best_dist.pth`: miglior distanza deterministica assoluta;
- `td3_det_best_dist_run.pth`: miglior distanza deterministica del run;
- `td3_expl_best_lap.pth`: giro valido in esplorazione;
- `td3_expl_best_dist.pth`: distanza esplorativa;
- `buffers/*.npz`: replay ed elite buffer.

Il commit su disco è ordinato:

1. salva buffer in `.npz`;
2. salva checkpoint temporaneo;
3. fsync;
4. rename atomico;
5. conserva `.bak` e `.prev`.

Al resume, se un buffer è più nuovo del checkpoint caricato, viene scartato a favore del backup allineato. Così non si riparte mai con pesi e replay appartenenti a due commit diversi.

---

## 13. Eval Deterministica

Ogni 5 episodi, dopo il warm-up, `td3_bc.py` esegue una valutazione senza rumore.

La eval usa:

- Actor in modalità deterministica;
- stessa mutual exclusion;
- stesso `gearing.compute_gear()`;
- stesso reset/relaunch TORCS.

I checkpoint `td3_det_*` sono quelli importanti per test e submission perché rappresentano la policy senza rumore. I checkpoint `td3_expl_*` restano utili per capire se l'esplorazione ha già trovato una traiettoria promettente.

---

## 14. Rollback e Actor Freezing

`./train_rl.sh --rollback` carica la migliore policy deterministica disponibile e congela temporaneamente l'Actor.

Durante il freeze:

- il Critic continua ad aggiornarsi;
- l'Actor resta fermo;
- la refinement automatica è sospesa.

Questo riallinea la value function alla policy ripristinata prima di riprendere gli aggiornamenti dell'Actor.

---

## 15. Refinement

La refinement serve quando le eval restano in plateau:

- `bc_weight` scende da `1.0` a `0.3`;
- il Critic non viene aggiornato;
- l'Actor segue una value function fissa;
- se supera stabilmente il plateau, si torna a `bc_weight=1.0`;
- se collassa sotto la soglia di sicurezza, si ricarica il miglior checkpoint deterministico.

Perché è separata: allentare sempre l'ancora BC renderebbe fragile il training normale. Farlo solo su plateau permette di spingere oltre un muro locale senza perdere il comportamento base.

---

## 16. Perché Questa Soluzione Ha Stabilizzato il Progetto

I problemi principali erano:

- covariate shift del BC;
- Actor che si allontanava troppo dai dati umani;
- Critic confuso da maschere terminali incoerenti;
- pedali simultanei;
- cambio marcia instabile;
- dataset sbilanciato da segmenti troppo concentrati;
- checkpoint e replay non sempre allineati dopo interruzioni.

La soluzione attuale funziona perché ogni problema ha un vincolo esplicito:

- covariate shift -> augmentation BC + stati online nel replay;
- drift Actor -> BC penalty su expert data;
- OOD online -> masking rigoroso della BC penalty;
- Critic confuso -> done masking coerente;
- pedali simultanei -> mutual exclusion moltiplicativa;
- cambio instabile -> `gearing.compute_gear()`;
- segmenti mirati -> esclusi dal BC, inclusi nel TD3 expert buffer;
- interruzioni -> checkpoint atomici con buffer allineati.

Il risultato è un sistema più piccolo e leggibile: una rete continua, una funzione di cambio, un wrapper ambiente, una pipeline dati.

---

## 17. Invarianti

Non modificare questi punti senza retrain e nuova validazione:

- stato rete 87D = tre frame 29D;
- `dist_from_start` solo metadato;
- BC solo su giri interi;
- TD3 expert buffer su giri + segmenti;
- Actor senza marcia predetta;
- `gearing.compute_gear()` in training/eval/test;
- `bc_weight=1.0` nel training normale;
- mutual exclusion moltiplicativa;
- checkpoint deterministici come sorgente per test e rollback.
