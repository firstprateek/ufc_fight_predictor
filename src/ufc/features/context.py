"""Context features: strength of schedule from the pre-fight plain Elo, activity and setting."""

from __future__ import annotations

from collections import defaultdict
from datetime import date

import numpy as np

from .bout import Bout, Card

NAN = float("nan")
ELO_START = 1500.0
ACTIVE_DAYS = 730  # "the last 24 months"

# Division order by weight limit (lb). Women's divisions share the ladder. Catch weight, open
# weight, tournaments and superfights have no place in it: their move is null.
WEIGHT_LIMIT = {
    "Strawweight": 115,
    "Women's Strawweight": 115,
    "Flyweight": 125,
    "Women's Flyweight": 125,
    "Bantamweight": 135,
    "Women's Bantamweight": 135,
    "Featherweight": 145,
    "Women's Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
}


def _results(s1: float | None) -> tuple[str, str]:
    if s1 is None:
        return "NC", "NC"
    if s1 == 0.5:
        return "D", "D"
    return ("W", "L") if s1 == 1.0 else ("L", "W")


class Context:
    def __init__(self):
        self.hist: dict[str, list[dict]] = defaultdict(list)

    def features(self, fid: str, on: date, card: Card) -> dict:
        h = self.hist[fid]
        cur = card.elo1 if fid == card.f1 else card.elo2
        limit = WEIGHT_LIMIT.get(card.weight_class)
        out = dict.fromkeys(
            [
                "ctx_sos_mean",
                "ctx_sos_win_mean",
                "ctx_best_win_elo",
                "ctx_last_opp_elo",
                "ctx_last_result",
                "ctx_last_finish",
                "ctx_elo_change3",
                "ctx_elo_peak_gap",
                "ctx_wc_move",
            ],
            NAN,
        )
        wins = [x["opp_elo"] for x in h if x["result"] == "W"]
        if h:
            out["ctx_sos_mean"] = float(np.mean([x["opp_elo"] for x in h]))
            out["ctx_last_opp_elo"] = h[-1]["opp_elo"]
            out["ctx_last_result"] = {"W": 1.0, "L": -1.0}.get(h[-1]["result"], 0.0)
            out["ctx_last_finish"] = float(h[-1]["finish"])
            out["ctx_elo_change3"] = cur - h[-min(3, len(h))]["own_pre"]
            out["ctx_elo_peak_gap"] = max(ELO_START, *(x["own_post"] for x in h)) - cur
            ranked = [x["limit"] for x in h if x["limit"] is not None]
            if limit is not None and ranked:
                out["ctx_wc_move"] = float(np.sign(limit - ranked[-1]))
        if wins:
            out["ctx_sos_win_mean"] = float(np.mean(wins))
            out["ctx_best_win_elo"] = float(max(wins))
        same = sum(x["wc"] == card.weight_class for x in h)
        out["ctx_fights_24m"] = float(sum((on - x["date"]).days <= ACTIVE_DAYS for x in h))
        out["ctx_wc_fights"] = float(same)
        out["ctx_wc_first"] = float(same == 0)
        out["ctx_five_round_fights"] = float(sum(x["five"] for x in h))
        out["ctx_title_fights"] = float(sum(x["title"] for x in h))
        out["ctx_is_five_rounds"] = (
            NAN if card.scheduled_rounds is None else float(card.scheduled_rounds == 5)
        )
        out["ctx_is_title"] = float(card.is_title)
        return out

    def pair(self, card: Card, on: date) -> dict:
        return {}

    def update(self, on: date, bouts: list[Bout], elo_after: dict) -> None:
        new = []
        for b in bouts:
            c = b.card
            res1, res2 = _results(b.s1)
            for fid, res, own_pre, opp_pre in (
                (c.f1, res1, c.elo1, c.elo2),
                (c.f2, res2, c.elo2, c.elo1),
            ):
                new.append(
                    (
                        fid,
                        {
                            "date": on,
                            "opp_elo": opp_pre,
                            "own_pre": own_pre,
                            "own_post": elo_after[fid],
                            "result": res,
                            "finish": b.finish,
                            "wc": c.weight_class,
                            "limit": WEIGHT_LIMIT.get(c.weight_class),
                            "five": c.scheduled_rounds == 5,
                            "title": c.is_title,
                        },
                    )
                )
        for fid, x in new:
            self.hist[fid].append(x)
