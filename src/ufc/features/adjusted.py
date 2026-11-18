"""Opponent-adjusted performance: an offence and a defence rating per fighter per dimension.

Per dimension, fighter A's observed value against B in a fight is modelled as
    league mean + sd * (offence_A - defence_B) + noise
with offence and defence in standard-deviation units (0 is the prior). A high defence means the
fighter suppresses the opponent's output. The mean and sd are the running mean and sd of every
observation before the date. After each date, for each observation with error
e = (value - mean) / sd - (offence_A - defence_B), offence_A moves by +rate_A * e and defence_B by
-rate_B * e, with rate = max(rate_min, 1 / (fights_with_stats + k0)) per fighter. Fights without
round stats update nothing. The update waits until MIN_OBS observations of a dimension exist, so
the first fights are not scaled by a meaningless sd.
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

from .bout import Bout, Card
from .groups import DIMENSIONS
from .params import RatingParams

MIN_OBS = 30


class _Running:
    """Welford running mean and (population) variance."""

    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def add(self, x: float) -> None:
        self.n += 1
        d = x - self.mean
        self.mean += d / self.n
        self.m2 += d * (x - self.mean)

    def scale(self) -> tuple[float, float]:
        """(mean, sd); sd is 1 until MIN_OBS observations exist (and if it is 0)."""
        if self.n < max(MIN_OBS, 2):
            return self.mean, 1.0
        sd = math.sqrt(self.m2 / self.n)
        return self.mean, sd if sd > 0 else 1.0


class Adjusted:
    def __init__(self, p: RatingParams):
        self.k0, self.rate_min = p.adj_k0, p.adj_rate_min
        self.off = {d: defaultdict(float) for d in DIMENSIONS}
        self.defn = {d: defaultdict(float) for d in DIMENSIONS}
        self.n: dict[str, int] = defaultdict(int)  # fights with stats
        self.run = {d: _Running() for d in DIMENSIONS}
        self.sq_err = {d: 0.0 for d in DIMENSIONS}  # one-step-ahead squared error (for tuning)
        self.n_err = {d: 0 for d in DIMENSIONS}
        self.score_from: date | None = None  # errors are recorded from this date on

    def _rate(self, fid: str) -> float:
        return max(self.rate_min, 1.0 / (self.n[fid] + self.k0))

    def features(self, fid: str, on: date, card: Card) -> dict:
        out = {}
        for d in DIMENSIONS:
            out[f"adj_{d}_off"] = self.off[d][fid]
            out[f"adj_{d}_def"] = self.defn[d][fid]
        return out

    def pair(self, card: Card, on: date) -> dict:
        """Expected value (in the dimension's own units) for each fighter in this matchup."""
        out = {}
        for d in DIMENSIONS:
            mean, sd = self.run[d].scale()
            e1 = mean + sd * (self.off[d][card.f1] - self.defn[d][card.f2])
            e2 = mean + sd * (self.off[d][card.f2] - self.defn[d][card.f1])
            out[f"adj_{d}_exp"] = (e1, e2)
        return out

    def update(self, on: date, bouts: list[Bout], elo_after=None) -> None:
        d_off = {d: defaultdict(float) for d in DIMENSIONS}
        d_def = {d: defaultdict(float) for d in DIMENSIONS}
        obs: list[tuple[str, float]] = []
        with_stats: list[str] = []
        for b in bouts:
            if not (b.m1 or b.m2):
                continue
            f1, f2 = b.card.f1, b.card.f2
            with_stats += [f1, f2]
            for a, o, vals in ((f1, f2, b.m1), (f2, f1, b.m2)):
                for d, y in vals.items():
                    obs.append((d, y))
                    if self.run[d].n < MIN_OBS:
                        continue
                    mean, sd = self.run[d].scale()
                    e = (y - mean) / sd - (self.off[d][a] - self.defn[d][o])
                    if self.score_from is None or on >= self.score_from:
                        self.sq_err[d] += e * e
                        self.n_err[d] += 1
                    d_off[d][a] += self._rate(a) * e
                    d_def[d][o] -= self._rate(o) * e
        for d in DIMENSIONS:
            for fid, x in d_off[d].items():
                self.off[d][fid] += x
            for fid, x in d_def[d].items():
                self.defn[d][fid] += x
        for d, y in obs:  # the league mean and sd see this date only after it
            self.run[d].add(y)
        for fid in with_stats:
            self.n[fid] += 1
