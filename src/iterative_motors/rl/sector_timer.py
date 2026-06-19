"""Telemetria a settori per la fase TIME-ATTACK: split times in stile gara.

Idea (presa dalla telemetria reale): la pista è divisa in N settori contigui sulla
``distFromStart``. Per ogni settore si tiene il MIGLIOR tempo mai percorso (best split),
persistito su un sidecar JSON così sopravvive ai restart. Durante il giro, alla chiusura di
ogni settore si premia (o penalizza) l'agente in base a quanto ha battuto il proprio record di
quel settore: è un segnale DENSO che dice esattamente DOVE guadagnare/perdere tempo.

Il "giro ideale teorico" è la somma dei migliori tempi-settore mai registrati — la linea
ottima assemblata pezzo per pezzo dai migliori parziali, anche provenienti da giri diversi.
Spingere l'agente a battere ogni settore lo avvicina progressivamente a quel limite.

Il modulo è puro (nessuna dipendenza da TORCS) per essere testabile in isolamento; l'unico
aggancio esterno è la scrittura atomica del sidecar tramite ``safe_write_text``.
"""

import os
import json

from ..common.checkpoint import safe_write_text


class SectorTimer:
    """Cronometro a settori con reward shaping e best-split persistiti.

    Uso nel training loop (solo time-attack):
      - ``start_lap(dist0)`` all'inizio dell'episodio;
      - ``reward += update(dist, cur_lap_time)`` ad ogni step;
      - ``finish_lap(lap_time)`` se il giro è completato (logga il breakdown), altrimenti
        ``discard_lap()`` (persiste i best parziali ma non logga il giro).
    """

    def __init__(self, track_length, n_sectors=18, sidecar_path=None,
                 reward_k=30.0, reward_cap=8.0, log=print):
        self.L = float(track_length)
        self.n = max(1, int(n_sectors))
        self.bounds = [self.L * i / self.n for i in range(self.n + 1)]
        self.reward_k = float(reward_k)
        self.reward_cap = float(reward_cap)
        self.sidecar_path = sidecar_path
        self._log = log
        self.best = [None] * self.n          # miglior durata per settore (s)
        self._dirty = False
        self._load()
        self.start_lap(0.0)

    # ── persistenza ────────────────────────────────────────────────────────
    def _load(self):
        if not self.sidecar_path or not os.path.exists(self.sidecar_path):
            return
        try:
            with open(self.sidecar_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if int(data.get('n_sectors', -1)) == self.n:
                best = data.get('best', [])
                if len(best) == self.n:
                    self.best = [None if b is None else float(b) for b in best]
                    have = sum(1 for b in self.best if b is not None)
                    self._log(f"  [TIME-ATTACK] Best-split caricati: {have}/{self.n} settori, "
                              f"giro ideale {self.ideal_lap():.3f}s.")
        except Exception as e:
            self._log(f"  [TIME-ATTACK] Impossibile leggere i best-split ({e}); riparto da zero.")

    def _save(self):
        if not self.sidecar_path or not self._dirty:
            return
        try:
            payload = json.dumps({'n_sectors': self.n, 'best': self.best})
            safe_write_text(self.sidecar_path, payload)
            self._dirty = False
        except Exception as e:
            self._log(f"  [TIME-ATTACK] Impossibile salvare i best-split: {e}")

    # ── ciclo del giro ─────────────────────────────────────────────────────
    def _sector_of(self, dist):
        """Indice del settore che contiene ``dist`` (clampato in [0, n-1])."""
        d = float(dist) % self.L
        idx = int(d / self.L * self.n)
        return max(0, min(self.n - 1, idx))

    def start_lap(self, dist0):
        """Reinizializza lo stato per un nuovo giro a partire da ``dist0`` (distFromStart)."""
        self.cur_sector = self._sector_of(dist0)
        self.sector_entry_time = 0.0
        self.prev_dist = None
        self.lap_sector_times = [None] * self.n

    def _commit_reward(self, i, duration):
        """Registra la durata del settore ``i``, calcola il premio vs record, aggiorna il best."""
        reward = 0.0
        self.lap_sector_times[i] = duration
        if self.best[i] is not None:
            delta = self.best[i] - duration                       # >0 se più veloce del record
            # Cap sul PREMIO (punti), non sul delta: limita gli spike a ±reward_cap per settore
            # (regione lineare ≈ ±reward_cap/reward_k secondi, oltre la quale satura).
            reward = max(-self.reward_cap, min(self.reward_cap, self.reward_k * delta))
            if duration < self.best[i]:
                self.best[i] = duration
                self._dirty = True
        else:
            self.best[i] = duration
            self._dirty = True
        return reward

    def update(self, dist, cur_lap_time):
        """Chiude i settori oltrepassati a questo step e ritorna la reward di split (0 se nessuno)."""
        if self.prev_dist is None:
            self.prev_dist = dist
            return 0.0
        reward = 0.0
        # Chiude ogni settore il cui confine superiore è stato superato (di norma 0 o 1 per step).
        while self.cur_sector < self.n - 1 and dist >= self.bounds[self.cur_sector + 1]:
            duration = cur_lap_time - self.sector_entry_time
            if duration > 0.0:
                reward += self._commit_reward(self.cur_sector, duration)
            self.sector_entry_time = cur_lap_time
            self.cur_sector += 1
        self.prev_dist = dist
        return reward

    def finish_lap(self, lap_time):
        """Chiude l'ultimo settore col tempo finale, logga il breakdown e persiste i best."""
        if lap_time and lap_time > self.sector_entry_time and self.cur_sector < self.n:
            duration = lap_time - self.sector_entry_time
            self._commit_reward(self.cur_sector, duration)
        self._log_breakdown(lap_time)
        self._save()

    def discard_lap(self):
        """Giro non completato: niente breakdown, ma persiste i best-settore già migliorati."""
        self._save()

    # ── diagnostica ────────────────────────────────────────────────────────
    def ideal_lap(self):
        """Somma dei migliori tempi-settore (limite teorico assemblato dai parziali ottimi)."""
        return sum(b for b in self.best if b is not None)

    def _log_breakdown(self, lap_time):
        ideal = self.ideal_lap()
        # Per ogni settore: quanto ha perso questo giro rispetto al proprio best-ever.
        losses = []
        for i, (d, b) in enumerate(zip(self.lap_sector_times, self.best)):
            if d is not None and b is not None:
                losses.append((d - b, i, d))
        losses.sort(reverse=True)
        worst = " ".join(f"S{i:02d}(+{dl:.2f}s)" for dl, i, _ in losses[:3] if dl > 1e-3)
        gap = (lap_time - ideal) if (lap_time and ideal > 0.0) else 0.0
        msg = (f"  [TIME-ATTACK] Giro {lap_time:.3f}s | ideale {ideal:.3f}s | gap {gap:+.3f}s")
        if worst:
            msg += f" | perde tempo: {worst}"
        self._log(msg)
