"""Rating hyperparameters. They are tuned only on fights dated before 2015-01-01 by
`ufc tune-ratings --before 2015-01-01` (see `tune.py`), which must reproduce these constants."""

from __future__ import annotations

from dataclasses import dataclass

TUNE_CUTOFF = "2015-01-01"  # the test period starts here; nothing at or after it may be tuned on
TUNE_SCORE_FROM = "2005-01-01"  # one-step-ahead error is scored on 2005-01-01 to the cutoff


@dataclass(frozen=True)
class RatingParams:
    glicko_c: float  # rating-deviation growth: rd^2 += c^2 * days since the last fight
    welo_k: float  # method-weighted Elo: K
    welo_unan: float  # multiplier for a unanimous decision (split / majority decision = 1)
    welo_ko: float  # KO / TKO / doctor's stoppage
    welo_sub: float  # submission
    welo_other: float  # DQ, overturned, could not continue, other
    adj_k0: float  # adjusted model: rate = max(adj_rate_min, 1 / (fights_with_stats + k0))
    adj_rate_min: float


# Tuned by `ufc tune-ratings --before 2015-01-01` (3,048 fights; one-step-ahead error 2005 to 2014).
RATING_PARAMS = RatingParams(
    glicko_c=6.0,
    welo_k=64.0,
    welo_unan=1.5,
    welo_ko=1.5,
    welo_sub=1.0,
    welo_other=0.5,
    adj_k0=12.0,
    adj_rate_min=0.08,
)
