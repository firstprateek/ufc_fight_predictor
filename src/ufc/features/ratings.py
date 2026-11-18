"""Glicko-1 and method-weighted Elo, as point-in-time trackers.

Both take one date at a time: every fight of a date is read from the state before the date, and the
updates apply after it (each date is one rating period).
"""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

from .bout import DECISION_SPLIT, DECISION_UNANIMOUS, KO, SUBMISSION, Bout, Card
from .params import RatingParams

START = 1500.0
RD_START = 350.0  # Glicko deviation of a debutant, and its cap
Q = math.log(10) / 400.0


def _g(rd: float) -> float:
    return 1.0 / math.sqrt(1.0 + 3.0 * Q * Q * rd * rd / math.pi**2)


def _expect(r: float, r_opp: float, rd_opp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-_g(rd_opp) * (r - r_opp) / 400.0))


class Glicko:
    """Glicko-1. The deviation of a fighter grows with the days since their last fight:
    rd = min(350, sqrt(rd_after_last_fight^2 + c^2 * days)). A debutant has rating 1500 and
    deviation 350 (the prior, a definition). A no contest moves nothing and does not reset the clock."""

    def __init__(self, c: float):
        self.c = c
        self.r: dict[str, float] = {}
        self.rd: dict[str, float] = {}
        self.last: dict[str, date] = {}

    def state(self, fid: str, on: date) -> tuple[float, float]:
        r, rd = self.r.get(fid, START), self.rd.get(fid, RD_START)
        if fid in self.last:
            rd = min(RD_START, math.sqrt(rd * rd + self.c * self.c * (on - self.last[fid]).days))
        return r, rd

    def expected(self, f1: str, f2: str, on: date) -> float:
        """Expected score of f1; both deviations enter (g of their root-sum-square)."""
        (r1, rd1), (r2, rd2) = self.state(f1, on), self.state(f2, on)
        return _expect(r1, r2, math.sqrt(rd1 * rd1 + rd2 * rd2))

    def features(self, fid: str, on: date, card: Card) -> dict:
        r, rd = self.state(fid, on)
        return {"glicko_r": r, "glicko_rd": rd}

    def pair(self, card: Card, on: date) -> dict:
        e = self.expected(card.f1, card.f2, on)
        return {"glicko_exp": (e, 1.0 - e)}

    def update(self, on: date, bouts: list[Bout], elo_after=None) -> None:
        games: dict[str, list[tuple[float, float, float]]] = defaultdict(list)
        for b in bouts:
            if b.s1 is None:
                continue
            f1, f2 = b.card.f1, b.card.f2
            (r1, rd1), (r2, rd2) = self.state(f1, on), self.state(f2, on)
            games[f1].append((r2, rd2, b.s1))
            games[f2].append((r1, rd1, 1.0 - b.s1))
        new = {}
        for fid, gl in games.items():
            r, rd = self.state(fid, on)
            d2_inv = (
                Q
                * Q
                * sum(_g(o) ** 2 * _expect(r, ro, o) * (1 - _expect(r, ro, o)) for ro, o, _ in gl)
            )
            denom = 1.0 / (rd * rd) + d2_inv
            new[fid] = (
                r + Q / denom * sum(_g(o) * (s - _expect(r, ro, o)) for ro, o, s in gl),
                math.sqrt(1.0 / denom),
            )
        for fid, (r, rd) in new.items():
            self.r[fid], self.rd[fid], self.last[fid] = r, rd, on


class MethodElo:
    """Elo whose update is K x multiplier(method) x (score - expected). Split and majority decisions
    have multiplier 1; the others are tuned. A draw uses the method's multiplier too."""

    def __init__(self, p: RatingParams):
        self.k = p.welo_k
        self.mult = {
            DECISION_SPLIT: 1.0,
            DECISION_UNANIMOUS: p.welo_unan,
            KO: p.welo_ko,
            SUBMISSION: p.welo_sub,
        }
        self.other = p.welo_other
        self.r: dict[str, float] = {}

    def rating(self, fid: str) -> float:
        return self.r.get(fid, START)

    def expected(self, f1: str, f2: str) -> float:
        return 1.0 / (1.0 + 10.0 ** ((self.rating(f2) - self.rating(f1)) / 400.0))

    def features(self, fid: str, on: date, card: Card) -> dict:
        return {"welo": self.rating(fid)}

    def pair(self, card: Card, on: date) -> dict:
        return {}

    def update(self, on: date, bouts: list[Bout], elo_after=None) -> None:
        delta: dict[str, float] = defaultdict(float)
        for b in bouts:
            if b.s1 is None:
                continue
            f1, f2 = b.card.f1, b.card.f2
            e = self.expected(f1, f2)
            step = self.k * self.mult.get(b.method, self.other) * (b.s1 - e)
            delta[f1] += step
            delta[f2] -= step
        for fid, d in delta.items():
            self.r[fid] = self.rating(fid) + d
