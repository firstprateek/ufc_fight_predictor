"""Historical betting odds: source CSV -> `odds.parquet`, matched to our `fights`.

The source file also holds the winner and many result-derived columns. Only the five columns in
`ODDS_COLUMNS` are ever read (design principle 7); `load_odds_source` is the single place the file
is opened.
"""

from __future__ import annotations

import csv
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .fetch import ODDS_FILE

ODDS_COLUMNS = ["R_fighter", "B_fighter", "R_odds", "B_odds", "date"]
OVERROUND_RANGE = (0.95, 1.25)
DATE_WINDOW_DAYS = 1
SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}
OUT_COLUMNS = ["fight_id", "odds_f1", "odds_f2", "p_market_f1", "overround"]


class OddsError(Exception):
    pass


def load_odds_source(path: Path) -> pd.DataFrame:
    """Read ONLY the five allowed columns of the odds CSV (by `usecols`, so no other column is ever
    parsed into memory). Rows with a missing or non-numeric price on either side are dropped by the
    caller via `clean_source`; here they are kept as NaN."""
    with path.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    missing = [c for c in ODDS_COLUMNS if c not in header]
    if missing:
        raise OddsError(f"odds file lacks columns {missing}")
    df = pd.read_csv(path, usecols=ODDS_COLUMNS, dtype={"R_fighter": str, "B_fighter": str})
    assert list(df.columns) == [c for c in header if c in ODDS_COLUMNS]
    return df[ODDS_COLUMNS]


def clean_source(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Keep rows with a usable American price on both sides (|odds| >= 100). Returns the rows and
    how many source rows had no usable odds."""
    ok = df.R_odds.notna() & df.B_odds.notna() & (df.R_odds.abs() >= 100) & (df.B_odds.abs() >= 100)
    return df[ok].reset_index(drop=True), int((~ok).sum())


def normalise_name(name: str) -> str:
    """Lower-case, accents and punctuation removed, hyphens as spaces, suffixes dropped."""
    s = unicodedata.normalize("NFKD", str(name))
    s = "".join(c for c in s if not unicodedata.combining(c)).lower()
    s = s.replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", "", s)
    parts = [p for p in s.split() if p not in SUFFIXES]
    return " ".join(parts)


def implied_prob(american) -> float:
    o = float(american)
    return 100.0 / (o + 100.0) if o > 0 else -o / (-o + 100.0)


def load_aliases(path: Path | None) -> dict[str, str]:
    """odds-file name -> our fighter name (normalised), from the committed overrides file."""
    if path is None or not path.exists():
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        return {
            normalise_name(r["odds_name"]): normalise_name(r["fighter_name"])
            for r in csv.DictReader(f)
        }


@dataclass
class OddsReport:
    source_rows: int = 0
    source_total: int = 0
    source_without_odds: int = 0
    matched: int = 0
    matched_by_fallback: int = 0
    fallback_matches: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    ambiguous: list[str] = field(default_factory=list)
    dropped_overround: list[str] = field(default_factory=list)
    per_year: dict[str, dict[str, int]] = field(default_factory=dict)
    scored_fights: int = 0
    scored_with_odds: int = 0


def _dates_near(d: str) -> list[str]:
    base = date.fromisoformat(d)
    return [
        (base + timedelta(days=k)).isoformat()
        for k in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1)
    ]


def match_odds(
    src: pd.DataFrame, fights: pd.DataFrame, fighters: pd.DataFrame, aliases: dict[str, str]
) -> tuple[list[dict], OddsReport, list[int]]:
    """Match cleaned odds rows to fights. Returns (rows, report, matched source row positions).
    Never guesses: a source row that fits no fight, or more than one, stays unmatched; a fight that
    two source rows claim is left unmatched for both."""
    rep = OddsReport(source_rows=len(src))
    name_of = dict(zip(fighters.fighter_id, fighters.name, strict=True))
    f = fights[["fight_id", "date", "fighter_1_id", "fighter_2_id"]].copy()
    f["n1"] = f.fighter_1_id.map(name_of).map(normalise_name)
    f["n2"] = f.fighter_2_id.map(name_of).map(normalise_name)
    by_pair: dict[frozenset, list[int]] = defaultdict(list)
    by_date: dict[str, list[int]] = defaultdict(list)
    for i, (n1, n2, d) in enumerate(zip(f.n1, f.n2, f.date, strict=True)):
        by_pair[frozenset((n1, n2))].append(i)
        by_date[d].append(i)

    def canon(n: str) -> str:
        n = normalise_name(n)
        return aliases.get(n, n)

    claims: dict[int, list[tuple[int, str]]] = defaultdict(list)  # fight row -> [(src pos, how)]
    for pos, r in enumerate(src.itertuples(index=False)):
        rn, bn = canon(r.R_fighter), canon(r.B_fighter)
        window = set(_dates_near(r.date))
        cands = [i for i in by_pair.get(frozenset((rn, bn)), []) if f.date.iat[i] in window]
        how = "exact"
        if not cands:
            cands = _fallback_candidates(rn, bn, by_date.get(r.date, []), f)
            how = "fallback"
        if len(cands) == 1:
            claims[cands[0]].append((pos, how))
        elif len(cands) > 1:
            rep.ambiguous.append(
                f"{r.date} {r.R_fighter} vs {r.B_fighter}: fits {len(cands)} fights"
            )
        else:
            rep.unmatched.append(f"{r.date} {r.R_fighter} vs {r.B_fighter}")

    rows, used = [], []
    for i, lst in sorted(claims.items()):
        if len(lst) > 1:
            for pos, _ in lst:
                r = src.iloc[pos]
                rep.ambiguous.append(
                    f"{r.date} {r.R_fighter} vs {r.B_fighter}: fight claimed by {len(lst)} odds rows"
                )
            continue
        pos, how = lst[0]
        r = src.iloc[pos]
        rn = canon(r.R_fighter)
        r_is_f1 = rn == f.n1.iat[i] or (rn != f.n2.iat[i] and _surname(rn) == _surname(f.n1.iat[i]))
        o1, o2 = (r.R_odds, r.B_odds) if r_is_f1 else (r.B_odds, r.R_odds)
        rows.append({"fight_id": f.fight_id.iat[i], "odds_f1": float(o1), "odds_f2": float(o2)})
        used.append(pos)
        if how == "fallback":
            rep.matched_by_fallback += 1
            rep.fallback_matches.append(
                f"{r.date} odds file: {r.R_fighter} vs {r.B_fighter}; ours: "
                f"{name_of[f.fighter_1_id.iat[i]]} vs {name_of[f.fighter_2_id.iat[i]]}"
            )
    return rows, rep, used


def _surname(n: str) -> str:
    return n.split()[-1] if n else ""


def _fallback_candidates(rn: str, bn: str, same_day: list[int], f: pd.DataFrame) -> list[int]:
    """Stricter second pass, same date only: the two fighters' surnames equal ours, and at least one
    fighter's full normalised name is identical (the other differs, e.g. Nick vs Nicholas)."""
    if _surname(rn) == _surname(bn):
        return []
    out = []
    for i in same_day:
        n1, n2 = f.n1.iat[i], f.n2.iat[i]
        for a, b in ((n1, n2), (n2, n1)):
            if _surname(a) == _surname(rn) and _surname(b) == _surname(bn):
                if (a == rn) != (b == bn):  # exactly one side identical
                    out.append(i)
                break
    return out


def make_odds_table(rows: list[dict], rep: OddsReport) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["fight_id", "odds_f1", "odds_f2"])
    q1 = df.odds_f1.map(implied_prob)
    q2 = df.odds_f2.map(implied_prob)
    df["overround"] = q1 + q2
    df["p_market_f1"] = q1 / (q1 + q2)
    lo, hi = OVERROUND_RANGE
    bad = ~df.overround.round(9).between(lo, hi)
    for r in df[bad].itertuples():
        rep.dropped_overround.append(
            f"{r.fight_id}: odds {r.odds_f1:+.0f} / {r.odds_f2:+.0f}, overround {r.overround:.3f}"
        )
    return df[~bad][OUT_COLUMNS].sort_values("fight_id").reset_index(drop=True)


def validate_odds(odds: pd.DataFrame, fights: pd.DataFrame) -> None:
    errors = []
    if not odds.fight_id.is_unique:
        errors.append("odds: more than one row for a fight")
    if not odds.fight_id.isin(fights.fight_id).all():
        errors.append("odds: fight_id missing from fights")
    if not ((odds.p_market_f1 > 0) & (odds.p_market_f1 < 1)).all():
        errors.append("odds: p_market_f1 outside (0, 1)")
    lo, hi = OVERROUND_RANGE
    if not odds.overround.round(9).between(lo, hi).all():
        errors.append(f"odds: overround outside [{lo}, {hi}]")
    if errors:
        raise OddsError("odds validation failed:\n- " + "\n- ".join(errors))


def build_odds(
    snapshot: Path,
    tables: dict[str, pd.DataFrame],
    aliases_path: Path | None = None,
    start: str = "2015-01-01",
) -> tuple[pd.DataFrame, OddsReport]:
    raw = load_odds_source(snapshot / ODDS_FILE)
    src, without = clean_source(raw)
    fights, fighters = tables["fights"], tables["fighters"]
    rows, rep, used = match_odds(src, fights, fighters, load_aliases(aliases_path))
    rep.source_total = len(raw)
    rep.source_without_odds = without
    odds = make_odds_table(rows, rep)
    validate_odds(odds, fights)
    rep.matched = len(odds)

    kept = set(odds.fight_id)
    matched_pos = {
        p for p, fid in zip(used, [r["fight_id"] for r in rows], strict=True) if fid in kept
    }
    years = raw.date.str[:4]
    with_odds_years = Counter(src.date.str[:4])
    used_years = Counter(src.date.iloc[sorted(used)].str[:4])
    matched_years = Counter(src.date.iloc[sorted(matched_pos)].str[:4])
    for y in sorted(set(years)):
        n = with_odds_years.get(y, 0)
        rep.per_year[y] = {
            "source": int((years == y).sum()),
            "with_odds": n,
            "matched": matched_years.get(y, 0),
            "dropped": used_years.get(y, 0) - matched_years.get(y, 0),
            "unmatched": n - used_years.get(y, 0),
        }
    scored = fights[(fights.outcome == "win") & (fights.date >= start)]
    rep.scored_fights = len(scored)
    rep.scored_with_odds = int(scored.fight_id.isin(kept).sum())
    return odds, rep
