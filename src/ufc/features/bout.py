"""One fight as the rating trackers see it, split so a feature can only read what is announced.

`Card` holds what is known before the fight (who, division, scheduled rounds, title). `Bout` adds
the outcome and the fight's own stats; it goes to trackers only for the update after the date.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import pandas as pd

from .groups import DIMENSIONS

DECISION_SPLIT = "split"  # split and majority decisions
DECISION_UNANIMOUS = "unanimous"
KO, SUBMISSION, OTHER = "ko", "sub", "other"


def method_category(method) -> str:
    """How the fight ended, for the method-weighted Elo."""
    if method in ("Decision - Split", "Decision - Majority"):
        return DECISION_SPLIT
    if method == "Decision - Unanimous":
        return DECISION_UNANIMOUS
    if method in ("KO/TKO", "TKO - Doctor's Stoppage"):
        return KO
    if method == "Submission":
        return SUBMISSION
    return OTHER


@dataclass(frozen=True)
class Card:
    """Known before the fight."""

    fight_id: str
    f1: str
    f2: str
    weight_class: str
    scheduled_rounds: float | None
    is_title: bool
    elo1: float  # plain Elo before the date (nan when the caller does not track it)
    elo2: float


@dataclass(frozen=True)
class Bout:
    """A decided-or-not fight with its outcome; used only after its date."""

    card: Card
    day: date
    s1: float | None  # score of fighter 1: 1 win, 0.5 draw, 0 loss; None for a no contest
    method: str  # method_category
    # DIMENSIONS value per fighter in this fight (own output), only where round stats allow
    m1: dict[str, float] = field(default_factory=dict)
    m2: dict[str, float] = field(default_factory=dict)

    @property
    def finish(self) -> bool:
        return self.method in (KO, SUBMISSION)


def dimension_values(metrics: dict) -> dict[str, float]:
    """DIMENSIONS values from `build._metrics` (num, den) pairs."""
    return {d: metrics[d][0] / metrics[d][1] for d in DIMENSIONS if d in metrics and metrics[d][1]}


def scheduled(x) -> float | None:
    return None if x is None or pd.isna(x) else float(x)
