"""Tune the rating hyperparameters on fights dated before the test period only.

`tune_ratings(..., before)` keeps fights dated strictly before `before` (never later than
`TUNE_CUTOFF`, 2015-01-01, where the test period starts) and scores every candidate by its
one-step-ahead error on 2005-01-01 to `before`: each fight is predicted from the state before its
date, then the state is updated. Grids are fixed here; the search is deterministic (grid search, or
coordinate descent where noted, ties keep the earlier grid value), so the committed constants in
`params.py` can be reproduced exactly by `ufc tune-ratings --before 2015-01-01`.
"""

from __future__ import annotations

import math
from dataclasses import replace
from datetime import date

import pandas as pd

from .adjusted import Adjusted
from .bout import Bout, Card, dimension_values, method_category, scheduled
from .build import _fight_totals, _metrics
from .params import TUNE_CUTOFF, TUNE_SCORE_FROM, RatingParams
from .ratings import Glicko, MethodElo

GLICKO_C = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 14.0)
WELO_K = (8.0, 12.0, 16.0, 24.0, 32.0, 48.0, 64.0, 96.0)
WELO_MULT = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
WELO_SWEEPS = 6
ADJ_K0 = (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0)
ADJ_RATE_MIN = (0.01, 0.02, 0.03, 0.05, 0.08, 0.12)


class TuneCutoffError(ValueError):
    """Tuning was asked to use fights from the test period."""


def pre_cutoff(fights: pd.DataFrame, stats: pd.DataFrame, before: str):
    """The fights (and their stats) dated strictly before `before`; refuses a later cutoff."""
    if before > TUNE_CUTOFF:
        raise TuneCutoffError(
            f"ratings are tuned only on fights before {TUNE_CUTOFF} (the test period), got {before}"
        )
    keep = fights[fights.date < before]
    assert keep.empty or keep.date.max() < before
    return keep, stats[stats.fight_id.isin(keep.fight_id)]


def make_days(fights: pd.DataFrame, stats: pd.DataFrame) -> list[tuple[date, list[Bout]]]:
    """Bouts grouped by date in order. Plain Elo is not tracked here (NaN on the cards)."""
    totals = _fight_totals(stats)
    nan = float("nan")
    days = []
    fights = fights.sort_values(["date", "event_id", "bout_order"], kind="stable")
    for day, group in fights.groupby("date", sort=True):
        on = date.fromisoformat(day)
        bouts = []
        for r in group.to_dict("records"):
            f1, f2 = r["fighter_1_id"], r["fighter_2_id"]
            if r["outcome"] == "win":
                s1 = 1.0 if r["winner_id"] == f1 else 0.0
            else:
                s1 = 0.5 if r["outcome"] == "draw" else None
            dur = None
            if pd.notna(r["end_round"]) and pd.notna(r["end_time_s"]):
                dur = (int(r["end_round"]) - 1) * 300 + int(r["end_time_s"])
            a, b = totals.get((r["fight_id"], f1)), totals.get((r["fight_id"], f2))
            m1 = dimension_values(_metrics(a, b, dur)) if a and b else {}
            m2 = dimension_values(_metrics(b, a, dur)) if a and b else {}
            card = Card(
                r["fight_id"],
                f1,
                f2,
                r["weight_class"],
                scheduled(r.get("scheduled_rounds")),
                bool(r.get("is_title", False)),
                nan,
                nan,
            )
            bouts.append(Bout(card, on, s1, method_category(r["method"]), m1, m2))
        days.append((on, bouts))
    return days


def _log_loss(tracker, days, score_from: date) -> float:
    """One-step-ahead log loss on decided fights from `score_from` on."""
    total, n = 0.0, 0
    for on, bouts in days:
        if on >= score_from:
            for b in bouts:
                if b.s1 in (0.0, 1.0):
                    e = tracker.predict(b.card, on)
                    p = e if b.s1 == 1.0 else 1.0 - e
                    total -= math.log(min(max(p, 1e-12), 1.0))
                    n += 1
        tracker.update(on, bouts)
    return total / n


class _GlickoPred(Glicko):
    def predict(self, card: Card, on: date) -> float:
        return self.expected(card.f1, card.f2, on)


class _WeloPred(MethodElo):
    def predict(self, card: Card, on: date) -> float:
        return self.expected(card.f1, card.f2)


def _tune_welo(days, score_from: date, p: RatingParams) -> tuple[RatingParams, float]:
    def loss(q: RatingParams) -> float:
        return _log_loss(_WeloPred(q), days, score_from)

    best, best_loss = p, loss(p)
    for _ in range(WELO_SWEEPS):
        moved = False
        for field, grid in (
            ("welo_k", WELO_K),
            ("welo_unan", WELO_MULT),
            ("welo_ko", WELO_MULT),
            ("welo_sub", WELO_MULT),
            ("welo_other", WELO_MULT),
        ):
            for v in grid:
                cand = replace(best, **{field: v})
                if cand == best:
                    continue
                cl = loss(cand)
                if cl < best_loss - 1e-12:
                    best, best_loss, moved = cand, cl, True
        if not moved:
            break
    return best, best_loss


def _adj_error(days, score_from: date, p: RatingParams) -> float:
    """Mean over dimensions of the one-step-ahead squared error in standard-deviation units."""
    a = Adjusted(p)
    a.score_from = score_from
    for on, bouts in days:
        a.update(on, bouts)
    return sum(a.sq_err[d] / a.n_err[d] for d in a.sq_err) / len(a.sq_err)


def tune_ratings(
    fights: pd.DataFrame,
    stats: pd.DataFrame,
    before: str = TUNE_CUTOFF,
    start: RatingParams | None = None,
) -> tuple[RatingParams, dict]:
    """Tuned constants and the diagnostics (errors at the optimum and at the untuned start)."""
    fights, stats = pre_cutoff(fights, stats, before)
    days = make_days(fights, stats)
    score_from = date.fromisoformat(TUNE_SCORE_FROM)
    base = start or RatingParams(0.0, 32.0, 1.0, 1.0, 1.0, 1.0, 5.0, 0.05)
    out = {"fights": len(fights), "scored_from": TUNE_SCORE_FROM, "before": before}

    glicko = {c: _log_loss(_GlickoPred(c), days, score_from) for c in GLICKO_C}
    c = min(glicko, key=lambda k: (glicko[k], k))
    out["glicko"] = {"c": c, "log_loss": glicko[c], "log_loss_c0": glicko[0.0]}

    welo_p, welo_loss = _tune_welo(days, score_from, replace(base, glicko_c=c))
    out["welo"] = {
        "log_loss": welo_loss,
        "log_loss_plain_elo": _log_loss(_WeloPred(base), days, score_from),
    }

    adj = {
        (k0, rm): _adj_error(days, score_from, replace(base, adj_k0=k0, adj_rate_min=rm))
        for k0 in ADJ_K0
        for rm in ADJ_RATE_MIN
    }
    k0, rm = min(adj, key=lambda k: (adj[k], k))
    out["adjusted"] = {
        "mse": adj[(k0, rm)],
        "mse_untuned": adj[(5.0, 0.05)],
        "k0": k0,
        "rate_min": rm,
    }
    return replace(welo_p, adj_k0=k0, adj_rate_min=rm), out
