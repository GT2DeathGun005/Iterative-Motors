"""Per-sector telemetry for the TIME-ATTACK phase: race-style split times.

Idea (taken from real telemetry): the track is divided into N contiguous sectors over
``distFromStart``. For each sector the BEST time ever driven (best split) is kept, persisted to
a JSON sidecar so it survives restarts. During the lap, at the close of each sector the agent is
rewarded (or penalized) based on how much it beat its own record for that sector: it is a DENSE
signal that says exactly WHERE time is gained/lost.

The "theoretical ideal lap" is the sum of the best sector times ever recorded — the optimal line
assembled piece by piece from the best splits, even coming from different laps. Pushing the agent
to beat every sector brings it progressively closer to that limit.

The module is pure (no TORCS dependency) so it is testable in isolation; the only external hook is
the atomic sidecar write via ``safe_write_text``.
"""

import os
import json

from ..common.checkpoint import safe_write_text


class SectorTimer:
    """Sector timer with reward shaping and persisted best splits.

    Use in the training loop (time-attack only):
      - ``start_lap(dist0)`` at the start of the episode;
      - ``reward += update(dist, cur_lap_time)`` at each step;
      - ``finish_lap(lap_time)`` if the lap is completed (logs the breakdown), otherwise
        ``discard_lap()`` (persists the partial bests but does not log the lap).
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
        self.best = [None] * self.n          # best duration per sector (s)
        self._dirty = False
        self._load()
        self.start_lap(0.0)

    # ── persistence ─────────────────────────────────────────────────────────
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

    # ── lap cycle ───────────────────────────────────────────────────────────
    def _sector_of(self, dist):
        """Index of the sector containing ``dist`` (clamped to [0, n-1])."""
        d = float(dist) % self.L
        idx = int(d / self.L * self.n)
        return max(0, min(self.n - 1, idx))

    def start_lap(self, dist0):
        """Reset the state for a new lap starting from ``dist0`` (distFromStart)."""
        self.cur_sector = self._sector_of(dist0)
        self.sector_entry_time = 0.0
        self.prev_dist = None
        self.lap_sector_times = [None] * self.n

    def _commit_reward(self, i, duration):
        """Record the duration of sector ``i``, compute the reward vs record, update the best."""
        reward = 0.0
        self.lap_sector_times[i] = duration
        if self.best[i] is not None:
            delta = self.best[i] - duration                       # >0 if faster than the record
            # Cap on the REWARD (points), not on the delta: limits spikes to ±reward_cap per sector
            # (linear region ≈ ±reward_cap/reward_k seconds, beyond which it saturates).
            reward = max(-self.reward_cap, min(self.reward_cap, self.reward_k * delta))
            if duration < self.best[i]:
                self.best[i] = duration
                self._dirty = True
        else:
            self.best[i] = duration
            self._dirty = True
        return reward

    def update(self, dist, cur_lap_time):
        """Close the sectors crossed at this step and return the split reward (0 if none)."""
        if self.prev_dist is None:
            self.prev_dist = dist
            return 0.0
        reward = 0.0
        # Close every sector whose upper bound has been crossed (normally 0 or 1 per step).
        while self.cur_sector < self.n - 1 and dist >= self.bounds[self.cur_sector + 1]:
            duration = cur_lap_time - self.sector_entry_time
            if duration > 0.0:
                reward += self._commit_reward(self.cur_sector, duration)
            self.sector_entry_time = cur_lap_time
            self.cur_sector += 1
        self.prev_dist = dist
        return reward

    def finish_lap(self, lap_time):
        """Close the last sector with the final time, log the breakdown and persist the bests."""
        if lap_time and lap_time > self.sector_entry_time and self.cur_sector < self.n:
            duration = lap_time - self.sector_entry_time
            self._commit_reward(self.cur_sector, duration)
        self._log_breakdown(lap_time)
        self._save()

    def discard_lap(self):
        """Lap not completed: no breakdown, but persist the already-improved sector bests."""
        self._save()

    # ── diagnostics ─────────────────────────────────────────────────────────
    def ideal_lap(self):
        """Sum of the best sector times (theoretical limit assembled from the optimal splits)."""
        return sum(b for b in self.best if b is not None)

    def _log_breakdown(self, lap_time):
        ideal = self.ideal_lap()
        # For each sector: how much this lap lost relative to its own best-ever.
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
