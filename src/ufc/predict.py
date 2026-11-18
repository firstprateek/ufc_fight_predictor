"""`ufc predict`: a win probability for a fight that has not happened yet.

The feature row of a proposed fight comes from the same `build_features` that builds the training
data: the fight is appended to `fights` as a row with no result (and no stats), so it is featurised
from the state before its date and never enters that state. The models are fitted by
`backtest.models.fit_predict`, the function the walk-forward backtest uses, on every decided fight
dated before the cutoff (the fight's date unless `train_before` says otherwise).
"""

from __future__ import annotations

import csv
import difflib
import json
import subprocess
import unicodedata
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest.models import LeakSuspected, fit_predict, flagged
from .backtest.protocol import FINAL_FEATURE_SET
from .backtest.run import scored_fights
from .features import (
    build_features,
    check_columns,
    check_physical,
    fit_imputation,
    parse_feature_set,
    swap_orientation,
)
from .odds import OVERROUND_RANGE, remove_margin

LOG_PATH = Path("predictions/log.csv")
LOG_COLUMNS = [
    "run_utc",
    "data_snapshot",
    "code_commit",
    "fighter_1_id",
    "fighter_1_name",
    "fighter_2_id",
    "fighter_2_name",
    "date",
    "weight_class",
    "scheduled_rounds",
    "is_title",
    "p_blend_1",
    "p_hgb_1",
    "p_logreg_1",
    "odds_1",
    "odds_2",
]
CARD_REQUIRED = ["fighter_1", "fighter_2"]
CARD_OPTIONAL = [
    "fighter_1_id",
    "fighter_2_id",
    "date",
    "weight_class",
    "scheduled_rounds",
    "is_title",
    "odds_1",
    "odds_2",
]

FEW_FIGHTS = 3
LAYOFF_DAYS = 730
STALE_DAYS = 14
FEW_TRAINING = 1000  # below this many training fights a warning is added
# From docs/results/ratings.md, "Against the market", full period, `all` features, `blend`.
MARKET_NOTE = (
    "Over 5,179 past fights the market's log loss was 0.6092 against 0.6474 for this model, "
    "so the market is the better estimate when the two disagree."
)

PROPOSED = "proposed-"  # fight_id prefix of the appended rows (real ids are hex)


class PredictError(Exception):
    """A request that cannot be answered; the message says why and what to do."""


@dataclass(frozen=True)
class Request:
    """A fight as typed: names, and anything the person chose to fix."""

    name1: str
    name2: str
    id1: str | None = None
    id2: str | None = None
    date: str | None = None
    weight_class: str | None = None
    rounds: int | None = None
    title: bool | None = None
    odds: tuple[float, float] | None = None


@dataclass(frozen=True)
class Matchup:
    """A resolved fight, in the order the person typed it."""

    id1: str
    id2: str
    name1: str
    name2: str
    date: str
    weight_class: str
    rounds: int
    title: bool
    gender: str
    odds: tuple[float, float] | None = None


# ---------------------------------------------------------------- names

_FOLD = str.maketrans({"ø": "o", "Ø": "O", "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ß": "ss"})


def norm_name(s) -> str:
    """Case, accents and runs of spaces ignored."""
    s = unicodedata.normalize("NFKD", str(s)).translate(_FOLD)
    s = "".join(c for c in s if not unicodedata.combining(c))
    return " ".join(s.casefold().split())


def _record(fights: pd.DataFrame, fid: str) -> tuple[int, int, int]:
    mine = fights[(fights.fighter_1_id == fid) | (fights.fighter_2_id == fid)]
    wins = int((mine.winner_id == fid).sum())
    losses = int(((mine.outcome == "win") & (mine.winner_id != fid)).sum())
    return wins, losses, len(mine) - wins - losses


def _last_fight(fights: pd.DataFrame, fid: str) -> pd.Series | None:
    mine = fights[(fights.fighter_1_id == fid) | (fights.fighter_2_id == fid)]
    if mine.empty:
        return None
    return mine.sort_values(["date", "bout_order"], kind="stable").iloc[-1]


def _describe(fights: pd.DataFrame, row) -> str:
    fid = row.fighter_id
    last = _last_fight(fights, fid)
    if last is None:
        return f"  {fid}  {row['name']}: no UFC fight"
    w, losses, o = _record(fights, fid)
    return (
        f"  {fid}  {row['name']}: UFC record {w}-{losses}-{o} (W-L-D/NC), "
        f"last fight {last.date}, {last.weight_class}"
    )


def resolve_fighter(
    name: str, fid: str | None, which: int, fighters: pd.DataFrame, fights: pd.DataFrame
) -> tuple[str, str]:
    """(fighter_id, name) for a typed name. Never guesses: no match and several matches both fail."""
    want = norm_name(name)
    if not want:
        raise PredictError(f"fighter {which}: the name is empty")
    if fid is not None:
        hit = fighters[fighters.fighter_id == fid]
        if hit.empty:
            raise PredictError(f"fighter {which}: no fighter has the id {fid!r}")
        if norm_name(hit.iloc[0]["name"]) != want:
            raise PredictError(
                f"fighter {which}: the id {fid} is {hit.iloc[0]['name']!r}, not {name!r}"
            )
        return fid, hit.iloc[0]["name"]
    normed = fighters["name"].map(norm_name)
    hits = fighters[normed == want]
    if len(hits) == 1:
        return hits.iloc[0].fighter_id, hits.iloc[0]["name"]
    if len(hits) > 1:
        flag = f"--id-{which}" if which in (1, 2) else "the id"
        lines = [_describe(fights, r) for _, r in hits.iterrows()]
        raise PredictError(
            f"fighter {which}: {len(hits)} fighters are called {name!r}; choose one with {flag} "
            f"(`fighter_{which}_id` in a card file):\n" + "\n".join(lines)
        )
    close = difflib.get_close_matches(want, sorted(set(normed)), n=5, cutoff=0.6)
    shown = sorted({fighters["name"][normed == c].iloc[0] for c in close})
    hint = f"closest names: {', '.join(shown)}" if shown else "no close names"
    raise PredictError(
        f"fighter {which}: no fighter called {name!r} ({hint}). If the fighter is new, "
        "`ufc fetch && ufc build` refreshes the data."
    )


def _check_date(value: str) -> str:
    try:
        ok = date.fromisoformat(value).isoformat() == value
    except ValueError:
        ok = False
    if not ok:
        raise PredictError(f"date must be YYYY-MM-DD, got {value!r}")
    return value


def _resolve_weight_class(name: str, fights: pd.DataFrame) -> str:
    known = sorted(set(fights.weight_class.dropna()))
    for k in known:
        if norm_name(k) == norm_name(name):
            return k
    raise PredictError(f"unknown weight class {name!r}; the data has: {', '.join(known)}")


def resolve(req: Request, tables: dict[str, pd.DataFrame], default_date: str) -> Matchup:
    """Names to ids, then the defaults for the bout (see design.md, milestone 4)."""
    fighters, fights = tables["fighters"], tables["fights"]
    day = _check_date(req.date or default_date)
    id1, n1 = resolve_fighter(req.name1, req.id1, 1, fighters, fights)
    id2, n2 = resolve_fighter(req.name2, req.id2, 2, fighters, fights)
    if id1 == id2:
        raise PredictError(f"{n1} is on both sides of the fight")
    past = fights[fights.date < day]
    last = [_last_fight(past, i) for i in (id1, id2)]
    if req.weight_class:
        wc = _resolve_weight_class(req.weight_class, fights)
    else:
        classes = [x.weight_class if x is not None else None for x in last]
        if classes[0] is not None and classes[0] == classes[1]:
            wc = classes[0]
        else:
            seen = ", ".join(
                f"{n} {c if c else 'no UFC fight before ' + day}"
                for n, c in zip((n1, n2), classes, strict=True)
            )
            raise PredictError(
                f"no default weight class ({seen}); pass --weight-class "
                f"(`weight_class` in a card file)"
            )
    title = bool(req.title)
    if title and req.rounds not in (None, 5):
        raise PredictError("a title fight is scheduled for 5 rounds; drop --rounds or use 5")
    rounds = 5 if title else (req.rounds if req.rounds is not None else 3)
    if rounds not in (3, 5):
        raise PredictError(f"rounds must be 3 or 5, got {rounds}")
    gender = next((x.gender for x in last if x is not None and pd.notna(x.gender)), None)
    if gender is None:
        gender = "female" if wc.startswith("Women's") else "male"
    return Matchup(id1, id2, n1, n2, day, wc, rounds, title, gender, _check_odds(req.odds))


def _check_odds(odds) -> tuple[float, float] | None:
    if odds is None:
        return None
    a, b = (float(x) for x in odds)
    if abs(a) < 100 or abs(b) < 100:
        raise PredictError(
            "odds are American prices such as -150 or +130 (never between -100 and 100)"
        )
    over = remove_margin(a, b)[2]
    lo, hi = OVERROUND_RANGE
    if not lo <= round(over, 9) <= hi:
        raise PredictError(
            f"odds {a:+.0f} / {b:+.0f} imply a total of {over:.3f}, outside {lo} to {hi}: "
            "check both prices"
        )
    return a, b


# ---------------------------------------------------------------- card file


def _boolean(s: str, where: str) -> bool:
    v = s.strip().lower()
    if v in ("1", "true", "yes", "y"):
        return True
    if v in ("0", "false", "no", "n", ""):
        return False
    raise PredictError(f"{where}: is_title must be true or false, got {s!r}")


def load_card(path: Path) -> list[Request]:
    """Read a card file. Blank optional cells mean "use the default"."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as e:
        raise PredictError(f"cannot read the card file: {e}") from e
    reader = csv.DictReader(text.splitlines())
    cols = [c.strip() for c in (reader.fieldnames or [])]
    missing = [c for c in CARD_REQUIRED if c not in cols]
    unknown = [c for c in cols if c not in CARD_REQUIRED + CARD_OPTIONAL]
    if missing:
        raise PredictError(f"the card file needs the columns {', '.join(CARD_REQUIRED)}")
    if unknown:
        raise PredictError(
            f"unknown card columns {unknown}; allowed: {', '.join(CARD_REQUIRED + CARD_OPTIONAL)}"
        )
    out = []
    for n, raw in enumerate(reader, start=2):  # row 1 is the header
        row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
        where = f"card row {n}"
        if not row["fighter_1"] or not row["fighter_2"]:
            raise PredictError(f"{where}: both fighter names are needed")
        try:
            rounds = int(row["scheduled_rounds"]) if row.get("scheduled_rounds") else None
            o1, o2 = row.get("odds_1", ""), row.get("odds_2", "")
            if bool(o1) != bool(o2):
                raise PredictError("give both odds_1 and odds_2, or neither")
            odds = (float(o1), float(o2)) if o1 else None
        except ValueError as e:
            raise PredictError(f"{where}: {e}") from e
        out.append(
            Request(
                row["fighter_1"],
                row["fighter_2"],
                row.get("fighter_1_id") or None,
                row.get("fighter_2_id") or None,
                row.get("date") or None,
                row.get("weight_class") or None,
                rounds,
                _boolean(row.get("is_title", ""), where) if row.get("is_title") else None,
                odds,
            )
        )
    if not out:
        raise PredictError("the card file has no fights")
    return out


# ---------------------------------------------------------------- features and models


def append_proposed(fights: pd.DataFrame, matchups: list[Matchup]) -> pd.DataFrame:
    """`fights` plus one result-less row per proposed fight (fighter_1_id < fighter_2_id, as stored)."""
    rows = []
    for i, m in enumerate(matchups):
        a, b = sorted((m.id1, m.id2))
        rows.append(
            {
                "fight_id": f"{PROPOSED}{i}",
                "event_id": f"{PROPOSED}{m.date}",
                "date": m.date,
                "bout_order": i + 1,
                "fighter_1_id": a,
                "fighter_2_id": b,
                "winner_id": None,
                "outcome": None,
                "method": None,
                "end_round": pd.NA,
                "end_time_s": pd.NA,
                "scheduled_rounds": m.rounds,
                "weight_class": m.weight_class,
                "is_title": m.title,
                "gender": m.gender,
                "referee": None,
            }
        )
    new = pd.DataFrame(rows).reindex(columns=fights.columns)
    for c in ("bout_order", "end_round", "end_time_s", "scheduled_rounds"):
        new[c] = new[c].astype("Int64")
    new["is_title"] = new["is_title"].astype(bool)
    return pd.concat([fights, new], ignore_index=True)


def _py(x):
    """A plain Python value (None for missing), so a result serialises to JSON."""
    if x is None or x is pd.NA or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, np.generic):
        x = x.item()
    return None if isinstance(x, float) and np.isnan(x) else x


def _side(row: pd.Series, side: str, fid: str, name: str, day: str) -> dict:
    g = lambda c: _py(row.get(f"{side}_{c}"))  # noqa: E731
    n, w, losses = int(g("n_fights")), int(g("n_wins") or 0), int(g("n_losses") or 0)
    days = g("days_since_last")
    return {
        "id": fid,
        "name": name,
        "ufc_fights": n,
        "ufc_record": {"wins": w, "losses": losses, "draws_nc": n - w - losses},
        "last_fight_date": (date.fromisoformat(day) - timedelta(days=int(days))).isoformat()
        if days is not None
        else None,
        "days_since_last_fight": None if days is None else int(days),
        "age": g("age"),
        "reach_cm": g("reach_cm"),
        "elo": g("elo"),
        "glicko": g("glicko_r"),
        "physical_imputed": bool(row[f"{side}_snapshot_imputed"]),
    }


def fair_american(p: float) -> str:
    """The American price with no margin for probability p."""
    if p >= 0.5:
        v = round(100 * p / (1 - p))
        return "+100" if v == 100 else f"-{v}"
    return f"+{round(100 * (1 - p) / p)}"


def _notes(sides: list[dict], day: str) -> list[str]:
    notes = []
    debut = [s["name"] for s in sides if s["ufc_fights"] == 0]
    if debut:
        notes.append("UFC debut: " + " and ".join(debut) + " (no prior UFC fight).")
    few = [f"{s['name']} ({s['ufc_fights']})" for s in sides if s["ufc_fights"] < FEW_FIGHTS]
    if few:
        notes.append(f"Fewer than {FEW_FIGHTS} prior UFC fights: " + ", ".join(few) + ".")
    imp = [s["name"] for s in sides if s["physical_imputed"]]
    if imp:
        notes.append(
            "Physical data filled in by a static rule (height, reach, stance or date of birth "
            "missing from the fighters table): " + ", ".join(imp) + "."
        )
    for s in sides:
        d = s["days_since_last_fight"]
        if d is not None and d > LAYOFF_DAYS:
            notes.append(
                f"Layoff over {LAYOFF_DAYS} days: {s['name']} last fought {d} days before {day}."
            )
    return notes


def _market(m: Matchup, p1: float) -> dict:
    a, b = m.odds
    q1, q2, over = remove_margin(a, b)
    return {
        "odds_1": a,
        "odds_2": b,
        "overround": over,
        "p_market_1": q1,
        "p_market_2": q2,
        "p_model_1": p1,
        "difference_1": p1 - q1,  # model minus market, in probability
        "note": MARKET_NOTE,
    }


def predict_matchups(
    matchups: list[Matchup],
    tables: dict[str, pd.DataFrame],
    *,
    groups: tuple[str, ...] | None = None,
    train_before: str | None = None,
) -> list[dict]:
    """One result per matchup, in order. Fights on one date are appended and featurised together,
    each from the state before that date. Training uses decided fights dated before `train_before`
    (default: the fight's date; never later than it)."""
    fights, fighters = tables["fights"], tables["fighters"]
    if groups is None:
        groups = parse_feature_set(FINAL_FEATURE_SET or "base")
    imp = fit_imputation(fights, fighters)  # fixed from the real tables, as in the training build
    results: list[dict | None] = [None] * len(matchups)
    for day in sorted({m.date for m in matchups}):
        idx = [i for i, m in enumerate(matchups) if m.date == day]
        group = [matchups[i] for i in idx]
        cutoff = train_before or day
        if cutoff > day:
            raise PredictError(f"training cutoff {cutoff} is after the fight date {day}")
        past = fights[fights.date < day]
        stats = tables["fight_stats_round"]
        feats = build_features(
            append_proposed(past, group),
            stats[stats.fight_id.isin(past.fight_id)],
            fighters,
            imp,
        )
        feats = feats.copy()  # build_features returns a fragmented frame
        check_physical(feats)
        check_columns(feats, groups)
        ids = [f"{PROPOSED}{k}" for k in range(len(group))]
        test = feats[feats.fight_id.isin(ids)].set_index("fight_id").loc[ids].reset_index()
        scored = scored_fights(feats[~feats.fight_id.isin(ids)], past)
        train = scored[scored.date < cutoff]
        keep = ~flagged(train)
        y = train.y.to_numpy()[keep]
        if len(y) == 0 or y.min() == y.max():
            raise PredictError(f"too few decided fights before {cutoff} to fit a model")
        assert train.empty or train.date.max() < min(cutoff, day), "training must precede the fight"
        if not swap_orientation(swap_orientation(test)).equals(test):
            raise LeakSuspected("swap_orientation is not an involution")
        p = fit_predict(train, train.y.to_numpy(), test, groups)
        p_swapped = fit_predict(train, train.y.to_numpy(), swap_orientation(test), groups)
        for name in ("hgb", "logreg"):  # the backtest's orientation guard
            if not np.allclose(p[name] + p_swapped[name], 1.0, atol=1e-9):
                raise LeakSuspected(f"{name}: swapping fighters did not give 1 - p")
        for k, (i, m) in enumerate(zip(idx, group, strict=True)):
            row = test.iloc[k]
            first = row.fighter_1_id == m.id1  # internal fighter 1 is the first-typed fighter
            p1 = {name: float(v[k] if first else 1.0 - v[k]) for name, v in p.items()}
            sides = [
                _side(row, "f1" if first else "f2", m.id1, m.name1, day),
                _side(row, "f2" if first else "f1", m.id2, m.name2, day),
            ]
            sides[0]["p_win"], sides[1]["p_win"] = p1["blend"], 1.0 - p1["blend"]
            for s in sides:
                s["fair_american_odds"] = fair_american(s["p_win"])
            results[i] = {
                "fighter_1": sides[0],
                "fighter_2": sides[1],
                "date": day,
                "weight_class": m.weight_class,
                "scheduled_rounds": m.rounds,
                "is_title": m.title,
                "probabilities_fighter_1": p1,
                "notes": _notes(sides, day),
                "market": _market(m, p1["blend"]) if m.odds else None,
                "fights_trained_on": int(len(y)),
                "training_cutoff": cutoff,
            }
    return results  # type: ignore[return-value]


def run(
    requests: list[Request],
    tables: dict[str, pd.DataFrame],
    snapshot: str,
    default_date: str | None = None,
    train_before: str | None = None,
) -> dict:
    """Resolve, predict, and wrap the results with the data footer. The document `--json` prints."""
    today = default_date or date.today().isoformat()
    matchups = []
    for n, req in enumerate(requests, start=1):
        try:
            matchups.append(resolve(req, tables, today))
        except PredictError as e:
            raise PredictError(f"fight {n}: {e}" if len(requests) > 1 else str(e)) from e
    fights = predict_matchups(matchups, tables, train_before=train_before)
    latest = tables["fights"].date.max()
    warnings = []
    for d in sorted({f["date"] for f in fights}):
        gap = (date.fromisoformat(d) - date.fromisoformat(latest)).days
        if gap > STALE_DAYS:
            warnings.append(
                f"The latest fight in the data is {latest}, {gap} days before {d}: fights since "
                "then are missing. `ufc fetch && ufc build` refreshes the data."
            )
    seen: dict[tuple[str, str], int] = {}
    for m in matchups:
        for fid, nm in ((m.id1, m.name1), (m.id2, m.name2)):
            seen[(m.date, fid)] = seen.get((m.date, fid), 0) + 1
            if seen[(m.date, fid)] == 2:
                warnings.append(f"{nm} is in more than one fight on {m.date}: check for a typo.")
    for d in sorted({f["date"] for f in fights}):
        n = min(f["fights_trained_on"] for f in fights if f["date"] == d)
        if n < FEW_TRAINING:
            warnings.append(
                f"Only {n} fights before {d} to train on: the probabilities are unreliable."
            )
    return {
        "fights": fights,
        "data": {
            "snapshot": snapshot,
            "latest_fight_date": latest,
            "fights_trained_on": sorted({f["fights_trained_on"] for f in fights}),
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------- output


def _pct(p: float) -> str:
    return f"{100 * p:.1f}%"


def _num(x, fmt: str) -> str:
    return "-" if x is None else format(x, fmt)


def render_fight(f: dict) -> str:
    a, b = f["fighter_1"], f["fighter_2"]
    w = max(len(a["name"]), len(b["name"]), 12)
    kind = "title fight, " if f["is_title"] else ""
    out = [
        f"{a['name']} vs {b['name']}",
        f"{f['date']}  |  {f['weight_class']}  |  {kind}{f['scheduled_rounds']} rounds",
        "",
    ]
    for s in (a, b):
        out.append(
            f"  {s['name']:<{w}}  {_pct(s['p_win']):>6}   fair price {s['fair_american_odds']}"
        )
    p = f["probabilities_fighter_1"]
    out += [
        f"  blend = mean of the logits of hgb and logistic regression; for {a['name']}: "
        f"hgb {_pct(p['hgb'])}, logistic regression {_pct(p['logreg'])}",
        "",
    ]

    def rec(s):
        r = s["ufc_record"]
        return f"{r['wins']}-{r['losses']}-{r['draws_nc']}"

    def last(s):
        d = s["days_since_last_fight"]
        return "none (debut)" if d is None else f"{s['last_fight_date']} ({d} d)"

    table = [
        ("UFC record (W-L-D/NC)", rec),
        ("Prior UFC fights", lambda s: str(s["ufc_fights"])),
        ("Last fight (days since)", last),
        ("Age", lambda s: _num(s["age"], ".1f")),
        ("Reach (cm)", lambda s: _num(s["reach_cm"], ".0f")),
        ("Elo", lambda s: _num(s["elo"], ".0f")),
        ("Glicko", lambda s: _num(s["glicko"], ".0f")),
    ]
    cells = [(label, fn(a), fn(b)) for label, fn in table]
    wa = max(len(a["name"]), *(len(c[1]) for c in cells))
    wb = max(len(b["name"]), *(len(c[2]) for c in cells))
    out.append(f"  {'':<24}  {a['name']:>{wa}}  {b['name']:>{wb}}")
    out += [f"  {label:<24}  {x:>{wa}}  {y:>{wb}}" for label, x, y in cells]
    if f["notes"]:
        out += ["", "Notes"] + [f"  - {n}" for n in f["notes"]]
    m = f["market"]
    if m:
        out += [
            "",
            f"Market ({m['odds_1']:+.0f} / {m['odds_2']:+.0f}; bookmaker margin removed, "
            f"total implied {m['overround']:.3f})",
        ]
        for s, pm, pq in (
            (a, m["p_market_1"], m["p_model_1"]),
            (b, m["p_market_2"], 1.0 - m["p_model_1"]),
        ):
            out.append(
                f"  {s['name']:<{w}}  market {_pct(pm):>6}   model {_pct(pq):>6}   "
                f"difference {100 * (pq - pm):+.1f} points"
            )
        out.append(f"  {m['note']}")
    return "\n".join(out)


def render(doc: dict) -> str:
    """The readable form of the document `--json` prints."""
    parts = ["\n\n".join(render_fight(f) for f in doc["fights"])]
    d = doc["data"]
    n = d["fights_trained_on"]
    trained = f"{n[0]:,}" if len(n) == 1 else f"{n[0]:,} to {n[-1]:,} (by date)"
    foot = [
        f"Data snapshot {d['snapshot']}; latest fight in the data {d['latest_fight_date']}; "
        f"trained on {trained} fights."
    ]
    foot += [f"WARNING: {w}" for w in doc["warnings"]]
    return parts[0] + "\n\n" + "\n".join(foot)


# ---------------------------------------------------------------- log


def code_commit() -> str:
    """The git commit of the code that ran (`-dirty` when it has uncommitted changes)."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=here,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def append_log(
    path: Path, doc: dict, commit: str | None = None, now: datetime | None = None
) -> int:
    """Append one row per fight to the prediction log (created with a header if absent). Existing
    rows are never touched; a file with a different header is refused rather than rewritten."""
    path = Path(path)
    now = now or datetime.now(UTC)
    commit = commit if commit is not None else code_commit()
    if path.exists() and path.stat().st_size:
        with path.open(newline="") as fh:
            header = next(csv.reader(fh), [])
        if header != LOG_COLUMNS:
            raise PredictError(f"{path} has a different header; move it aside and run again")
        fresh = False
    else:
        fresh = True
    path.parent.mkdir(parents=True, exist_ok=True)
    glued = not fresh and not path.read_bytes().endswith(b"\n")
    with path.open("a", newline="") as fh:
        if glued:  # an unterminated last line must not swallow the first new row
            fh.write("\n")
        w = csv.writer(fh)
        if fresh:
            w.writerow(LOG_COLUMNS)
        for f in doc["fights"]:
            a, b, p, m = f["fighter_1"], f["fighter_2"], f["probabilities_fighter_1"], f["market"]
            w.writerow(
                [
                    now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    doc["data"]["snapshot"],
                    commit,
                    a["id"],
                    a["name"],
                    b["id"],
                    b["name"],
                    f["date"],
                    f["weight_class"],
                    f["scheduled_rounds"],
                    f["is_title"],
                    repr(p["blend"]),
                    repr(p["hgb"]),
                    repr(p["logreg"]),
                    "" if m is None else f"{m['odds_1']:g}",
                    "" if m is None else f"{m['odds_2']:g}",
                ]
            )
    return len(doc["fights"])


def to_json(doc: dict) -> str:
    return json.dumps(doc, indent=2, default=_py)
