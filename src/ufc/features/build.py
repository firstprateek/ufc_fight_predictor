"""Build `features.parquet` from the milestone 1 tables.

Point-in-time rule: the features of a fight use only fights dated strictly before it. Fights are
processed one date at a time: every fight of a date is featurised from the state as it stood before
that date, and only then is the state updated. Same-day earlier bouts are ignored on purpose.

`fighters` is a current snapshot, so only height, reach, stance and date of birth are used from it.
Whether those are filled in depends on how long the fighter lasted in the UFC, so their missingness
reveals the future: it is imputed away with static rules (see `Imputation`) and never reaches a model.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from .adjusted import Adjusted
from .bout import Bout, Card, dimension_values, method_category, scheduled
from .context import Context
from .groups import in_set
from .params import RATING_PARAMS, RatingParams
from .ratings import Glicko, MethodElo

ELO_START = 1500.0
ELO_K = 32.0
# Exponentially weighted rates: a prior fight's weight halves for every 3 fights further back
# (most recent prior fight has weight 1, the one before 0.5 ** (1/3), ...).
HALF_LIFE_FIGHTS = 3.0

# metric -> description. Each fight gives (numerator, denominator) per metric; a rate is
# sum(w * num) / sum(w * den) over prior fights, null when the denominator is 0 or no fight has it.
METRICS = [
    "slpm",  # significant strikes landed per minute
    "sapm",  # significant strikes absorbed per minute
    "str_acc",  # significant strike accuracy
    "str_def",  # share of opponent's significant strike attempts that missed
    "td15",  # takedowns landed per 15 minutes
    "td_acc",
    "td_def",
    "sub15",  # submission attempts per 15 minutes
    "kd15",  # knockdowns scored per 15 minutes
    "kda15",  # knockdowns suffered per 15 minutes
    "ctrl_share",  # share of fight time in control
    "ctrled_share",  # share of fight time being controlled
]
STATIC = ["height_cm", "reach_cm"]
PHYSICAL = ["age", "height_cm", "reach_cm", "stance", "southpaw"]
IMPUTED_STANCE = "Orthodox"
# Per fighter, true when any of height, reach, stance or dob was filled in. A bookkeeping column for
# the backtest (it keeps such fights out of training); it is never a model input.
FLAG = "snapshot_imputed"
KEYS = ["fight_id", "event_id", "date", "fighter_1_id", "fighter_2_id"]

_SUMMED = [
    "sig_str_landed",
    "sig_str_att",
    "td_landed",
    "td_att",
    "sub_attempts",
    "knockdowns",
    "control_s",
]


def _method_class(method) -> str:
    if method in ("KO/TKO", "TKO - Doctor's Stoppage"):
        return "ko"
    if method == "Submission":
        return "sub"
    return "other"


def _fight_totals(stats: pd.DataFrame) -> dict[tuple[str, str], dict]:
    """(fight_id, fighter_id) -> summed round stats. control_s is null only if null in every round."""
    if stats.empty:
        return {}
    g = stats.groupby(["fight_id", "fighter_id"])[_SUMMED].sum(min_count=1)
    return {k: {c: v[c] for c in _SUMMED} for k, v in g.to_dict("index").items()}


def _num(x):
    return None if x is None or pd.isna(x) else float(x)


def _metrics(own: dict, opp: dict, duration_s: float | None) -> dict:
    """Per-metric (numerator, denominator) for one fighter in one fight."""
    out: dict[str, tuple[float, float]] = {}
    sig_l, sig_a = _num(own["sig_str_landed"]), _num(own["sig_str_att"])
    osig_l, osig_a = _num(opp["sig_str_landed"]), _num(opp["sig_str_att"])
    td_l, td_a = _num(own["td_landed"]), _num(own["td_att"])
    otd_l, otd_a = _num(opp["td_landed"]), _num(opp["td_att"])
    if sig_a is not None:
        out["str_acc"] = (sig_l, sig_a)
    if osig_a is not None:
        out["str_def"] = (osig_a - osig_l, osig_a)
    if td_a is not None:
        out["td_acc"] = (td_l, td_a)
    if otd_a is not None:
        out["td_def"] = (otd_a - otd_l, otd_a)
    if duration_s:
        mins = duration_s / 60.0
        out["slpm"] = (sig_l, mins)
        out["sapm"] = (osig_l, mins)
        out["td15"] = (td_l, mins / 15.0)
        out["sub15"] = (_num(own["sub_attempts"]), mins / 15.0)
        out["kd15"] = (_num(own["knockdowns"]), mins / 15.0)
        out["kda15"] = (_num(opp["knockdowns"]), mins / 15.0)
        if _num(own["control_s"]) is not None:
            out["ctrl_share"] = (_num(own["control_s"]), duration_s)
        if _num(opp["control_s"]) is not None:
            out["ctrled_share"] = (_num(opp["control_s"]), duration_s)
    return {k: v for k, v in out.items() if None not in v}


def _rate(hist: list[dict], metric: str, half_life: float | None) -> float | None:
    """Ratio of (weighted) sums over prior fights. half_life None = plain career ratio."""
    num = den = 0.0
    seen = False
    n = len(hist)
    for i, h in enumerate(hist):
        m = h["metrics"].get(metric)
        if m is None:
            continue
        w = 1.0 if half_life is None else 0.5 ** ((n - 1 - i) / half_life)
        num += w * m[0]
        den += w * m[1]
        seen = True
    return num / den if seen and den > 0 else None


def _history_features(hist: list[dict], on: date) -> dict:
    """Features of a fighter from their prior fights (chronological list, all strictly before `on`)."""
    n = len(hist)
    out: dict = {"n_fights": n}
    names = ["n_wins", "n_losses", "win_rate", "win_streak", "loss_streak", "days_since_last"]
    names += ["ko_win_share", "sub_win_share", "ko_loss_share"]
    names += [f"{m}_{kind}" for m in METRICS for kind in ("career", "ewm")]
    out.update(dict.fromkeys(names))
    if n == 0:
        return out
    res = [h["result"] for h in hist]
    wins, losses = res.count("W"), res.count("L")
    out["n_wins"], out["n_losses"] = wins, losses
    out["win_rate"] = wins / n
    for key, target in (("win_streak", "W"), ("loss_streak", "L")):
        k = 0
        for r in reversed(res):
            if r != target:
                break
            k += 1
        out[key] = k
    out["days_since_last"] = (on - hist[-1]["date"]).days
    if wins:
        out["ko_win_share"] = sum(h["result"] == "W" and h["method"] == "ko" for h in hist) / wins
        out["sub_win_share"] = sum(h["result"] == "W" and h["method"] == "sub" for h in hist) / wins
    if losses:
        out["ko_loss_share"] = (
            sum(h["result"] == "L" and h["method"] == "ko" for h in hist) / losses
        )
    for m in METRICS:
        out[f"{m}_career"] = _rate(hist, m, None)
        out[f"{m}_ewm"] = _rate(hist, m, HALF_LIFE_FIGHTS)
    return out


class PhysicalNullError(ValueError):
    """A physical column holds a null: snapshot missingness would reach a model."""


@dataclass(frozen=True)
class Imputation:
    """Static rules that fill the snapshot fields. They use fight dates and divisions, never results.

    height: median of fighters in the same (weight class, gender), else the gender median, else the
    global median. reach: the fighter's (possibly imputed) height x the global median reach/height
    ratio. stance: `Orthodox`. age: the global median age at fight time.
    """

    height_global: float
    height_gender: dict
    height_group: dict
    reach_ratio: float
    age: float
    stance: str
    group_of: dict  # fighter_id -> (weight_class, gender), the fighter's most frequent division

    def height(self, fid) -> float:
        group = self.group_of.get(fid)
        if group in self.height_group:
            return self.height_group[group]
        if group is not None and group[1] in self.height_gender:
            return self.height_gender[group[1]]
        return self.height_global


def fit_imputation(fights: pd.DataFrame, fighters: pd.DataFrame) -> Imputation:
    """Imputation constants from the full tables. Fixed for a build, so a truncated rebuild (the
    leak check) uses the same values."""
    long = pd.concat(
        [
            fights[["fighter_1_id", "weight_class", "gender", "date"]].rename(
                columns={"fighter_1_id": "fid"}
            ),
            fights[["fighter_2_id", "weight_class", "gender", "date"]].rename(
                columns={"fighter_2_id": "fid"}
            ),
        ]
    )
    group_of = {}
    for fid, g in long.groupby("fid"):
        counts = g.groupby(["weight_class", "gender"]).size().sort_index()
        group_of[fid] = counts.idxmax()  # most frequent division, ties by name
    fx = fighters.set_index("fighter_id")
    h = fx["height_cm"].dropna().astype(float)
    gh = pd.Series({f: group_of[f] for f in h.index if f in group_of})
    hh = h[gh.index]
    height_group = hh.groupby([gh.map(lambda t: t[0]), gh.map(lambda t: t[1])]).median().to_dict()
    height_gender = hh.groupby(gh.map(lambda t: t[1])).median().to_dict()
    both = fx[["height_cm", "reach_cm"]].dropna().astype(float)
    ratio = float((both.reach_cm / both.height_cm).median())
    dob = pd.to_datetime(long.fid.map(fx["dob"]), errors="coerce")
    ages = (pd.to_datetime(long.date) - dob).dt.days / 365.25
    return Imputation(
        height_global=float(h.median()),
        height_gender={k: float(v) for k, v in height_gender.items()},
        height_group={k: float(v) for k, v in height_group.items()},
        reach_ratio=ratio,
        age=float(ages.dropna().median()),
        stance=IMPUTED_STANCE,
        group_of=group_of,
    )


def _static_features(fid, fighter: dict | None, on: date, imp: Imputation) -> dict:
    """Physical features with every missing snapshot field imputed (never null)."""
    f = fighter or {}
    imputed = any(
        f.get(k) is None or pd.isna(f.get(k)) for k in ("height_cm", "reach_cm", "stance", "dob")
    )
    dob = f.get("dob")
    if dob is not None and not pd.isna(dob):
        age = (on - date.fromisoformat(dob)).days / 365.25
    else:
        age = imp.age
    height = _num(f.get("height_cm"))
    if height is None:
        height = imp.height(fid)
    reach = _num(f.get("reach_cm"))
    if reach is None:
        reach = height * imp.reach_ratio
    stance = f.get("stance")
    stance = imp.stance if stance is None or pd.isna(stance) else stance
    return {
        "age": age,
        "height_cm": height,
        "reach_cm": reach,
        "stance": stance,
        "southpaw": float(stance == "Southpaw"),
        FLAG: imputed,
    }


def check_physical(df: pd.DataFrame) -> None:
    """Raise if any physical column (per fighter or diff) has a null."""
    cols = [f"{p}_{c}" for p in ("f1", "f2", "diff") for c in PHYSICAL if f"{p}_{c}" in df.columns]
    bad = [c for c in cols if df[c].isna().any()]
    if bad:
        raise PhysicalNullError(f"null in physical columns {bad}: snapshot missingness would leak")


def _elo_expected(r1: float, r2: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((r2 - r1) / 400.0))


def feature_columns(df: pd.DataFrame, groups: tuple[str, ...] | None = None) -> list[str]:
    """Numeric model inputs: every f1_/f2_/diff_ column except the stance strings and the
    snapshot_imputed flags (which must never reach a model). `groups` (the groups on besides
    `base`) restricts the columns to a feature set; None keeps every column."""
    skip = ("f1_stance", "f2_stance", f"f1_{FLAG}", f"f2_{FLAG}")
    cols = [c for c in df.columns if c.startswith(("f1_", "f2_", "diff_")) and c not in skip]
    return cols if groups is None else [c for c in cols if in_set(c, groups)]


def swap_orientation(df: pd.DataFrame) -> pd.DataFrame:
    """The same fights seen from the other fighter: f1_ <-> f2_, diff_ negated, ids swapped."""
    out = df.copy()
    for c in df.columns:
        if c.startswith("f1_"):
            out[c], out["f2_" + c[3:]] = df["f2_" + c[3:]].array, df[c].array
        elif c.startswith("diff_"):
            out[c] = -df[c]
    if "fighter_1_id" in df.columns:
        out["fighter_1_id"], out["fighter_2_id"] = df["fighter_2_id"], df["fighter_1_id"]
    return out


def build_features(
    fights: pd.DataFrame,
    stats: pd.DataFrame,
    fighters: pd.DataFrame,
    imputation: Imputation | None = None,
    params: RatingParams = RATING_PARAMS,
) -> pd.DataFrame:
    """One row per fight: KEYS, `f1_*`, `f2_*` and `diff_*` (= f1 minus f2) columns.
    `imputation` defaults to constants fitted on `fights` and `fighters`; `params` are the tuned
    rating constants."""
    imp = imputation or fit_imputation(fights, fighters)
    trackers = [Glicko(params.glicko_c), MethodElo(params), Adjusted(params), Context()]
    fights = fights.sort_values(["date", "event_id", "bout_order"], kind="stable")
    totals = _fight_totals(stats)
    static = fighters.set_index("fighter_id").to_dict("index")
    history: dict[str, list[dict]] = defaultdict(list)
    elo: dict[str, float] = defaultdict(lambda: ELO_START)
    rows: list[dict] = []

    for day, group in fights.groupby("date", sort=True):
        on = date.fromisoformat(day)
        recs = group.to_dict("records")
        pre_elo = {}
        cards = {}
        for r in recs:  # 1. features from the state before this date
            row = {k: r[k] for k in KEYS}
            parts = {}
            card = Card(
                r["fight_id"],
                r["fighter_1_id"],
                r["fighter_2_id"],
                r["weight_class"],
                scheduled(r.get("scheduled_rounds")),
                bool(r.get("is_title", False)),
                elo[r["fighter_1_id"]],
                elo[r["fighter_2_id"]],
            )
            cards[r["fight_id"]] = card
            for side, fid in (("f1", r["fighter_1_id"]), ("f2", r["fighter_2_id"])):
                f = {
                    **_history_features(history[fid], on),
                    **_static_features(fid, static.get(fid), on, imp),
                }
                f["elo"] = elo[fid]
                for t in trackers:
                    f.update(t.features(fid, on, card))
                row[f"{side}_{FLAG}"] = f.pop(FLAG)
                parts[side] = f
                row.update({f"{side}_{k}": v for k, v in f.items()})
            for t in trackers:
                for k, (v1, v2) in t.pair(card, on).items():
                    parts["f1"][k], parts["f2"][k] = v1, v2
                    row[f"f1_{k}"], row[f"f2_{k}"] = v1, v2
            for k in parts["f1"]:
                if k == "stance":
                    continue
                a, b = parts["f1"][k], parts["f2"][k]
                row[f"diff_{k}"] = np.nan if a is None or b is None else a - b
            rows.append(row)
            pre_elo[r["fight_id"]] = (elo[r["fighter_1_id"]], elo[r["fighter_2_id"]])
        new_hist, delta, bouts = [], defaultdict(float), []
        for r in recs:  # 2. update the state with this date's fights
            f1, f2 = r["fighter_1_id"], r["fighter_2_id"]
            if pd.isna(r["outcome"]):
                continue  # a proposed fight (`ufc predict`): featurised above, never enters the state
            if r["outcome"] == "win":
                assert r["winner_id"] in (f1, f2), f"winner of {r['fight_id']} is not a fighter"
                res1, res2, s1 = ("W", "L", 1.0) if r["winner_id"] == f1 else ("L", "W", 0.0)
            elif r["outcome"] == "draw":
                res1 = res2 = "D"
                s1 = 0.5
            else:  # no contest: counts as a fight, moves no rating
                res1 = res2 = "NC"
                s1 = None
            if s1 is not None:
                e1 = _elo_expected(*pre_elo[r["fight_id"]])
                delta[f1] += ELO_K * (s1 - e1)
                delta[f2] += ELO_K * ((1.0 - s1) - (1.0 - e1))
            dur = None
            if pd.notna(r["end_round"]) and pd.notna(r["end_time_s"]):
                dur = (int(r["end_round"]) - 1) * 300 + int(r["end_time_s"])
            a, b = totals.get((r["fight_id"], f1)), totals.get((r["fight_id"], f2))
            m1 = _metrics(a, b, dur) if a and b else {}
            m2 = _metrics(b, a, dur) if a and b else {}
            method = _method_class(r["method"])
            bouts.append(
                Bout(
                    cards[r["fight_id"]],
                    on,
                    s1,
                    method_category(r["method"]),
                    dimension_values(m1),
                    dimension_values(m2),
                )
            )
            new_hist.append((f1, {"date": on, "result": res1, "method": method, "metrics": m1}))
            new_hist.append((f2, {"date": on, "result": res2, "method": method, "metrics": m2}))
        for fid, h in new_hist:
            history[fid].append(h)
        for fid, d in delta.items():
            elo[fid] += d
        for t in trackers:
            t.update(on, bouts, elo)

    df = pd.DataFrame(rows)
    ints = [c for c in df.columns if c.endswith(("n_fights", "n_wins", "n_losses"))]
    ints += [c for c in df.columns if c.endswith(("win_streak", "loss_streak", "days_since_last"))]
    for c in df.columns:
        if c in ints:
            df[c] = df[c].astype("Int64")
        elif c not in KEYS and c not in ("f1_stance", "f2_stance", f"f1_{FLAG}", f"f2_{FLAG}"):
            df[c] = df[c].astype("float64")
    return df.reset_index(drop=True)


def check_truncation(
    fights: pd.DataFrame,
    stats: pd.DataFrame,
    fighters: pd.DataFrame,
    full: pd.DataFrame,
    sample: int = 200,
    seed: int = 0,
) -> list[str]:
    """Leak guard. For a random sample of fights, rebuild the features with every fight dated on or
    after the sampled fight removed (except the fight itself) and require an identical row.
    Returns the fight_ids that differ."""
    imp = fit_imputation(fights, fighters)  # fixed from the full tables, as in the full build
    ids = sorted(fights.fight_id)
    picks = random.Random(seed).sample(ids, min(sample, len(ids)))
    by_id = fights.set_index("fight_id")
    full = full.set_index("fight_id")
    bad = []
    for fid in picks:
        d = by_id.at[fid, "date"]
        keep = fights[(fights.date < d) | (fights.fight_id == fid)]
        part = build_features(keep, stats[stats.fight_id.isin(keep.fight_id)], fighters, imp)
        a = part.set_index("fight_id").loc[fid]
        b = full.loc[fid]
        same = all((pd.isna(a[c]) and pd.isna(b[c])) or a[c] == b[c] for c in full.columns)
        if not same:
            bad.append(fid)
    return bad
