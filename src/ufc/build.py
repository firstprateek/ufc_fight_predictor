"""Source CSVs -> validated Parquet tables."""

from __future__ import annotations

import collections
import csv
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

from . import parse as P
from .validate import ValidationError, validate

STAT_PAIRS = {
    "sig_str": "SIG.STR.",
    "total_str": "TOTAL STR.",
    "td": "TD",
    "head": "HEAD",
    "body": "BODY",
    "leg": "LEG",
    "distance": "DISTANCE",
    "clinch": "CLINCH",
    "ground": "GROUND",
}


@dataclass
class BuildReport:
    source_fights: int = 0
    dropped: list[tuple[int, str]] = field(default_factory=list)  # (rows, reason)
    unresolved: list[str] = field(default_factory=list)
    ambiguous_stats_fights: int = 0
    counts: dict[str, int] = field(default_factory=dict)
    fights_without_round_stats: int = 0
    resolved_by_rule: int = 0
    resolved_by_override: int = 0


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_overrides(path: Path | None) -> dict[str, str]:
    """name -> fighter_id, from the committed overrides file. An override wins over the weight/age
    rule, and also covers names the fight lists spell differently from the fighter pages."""
    if path is None or not path.exists():
        return {}
    return {r["name"]: r["fighter_id"] for r in _read(path)}


def load_dropped(path: Path | None) -> dict[str, str]:
    """fight_id -> reason, for fights we refuse to guess a fighter for."""
    if path is None or not path.exists():
        return {}
    return {r["fight_id"]: r["reason"] for r in _read(path)}


class NameIndex:
    """Fighter name -> candidate ids, with the rules for choosing among duplicates."""

    def __init__(self, fighters: dict[str, dict], names: dict[str, set[str]], overrides: dict):
        self.fighters, self.names, self.overrides = fighters, names, overrides
        self.by_rule = 0
        self.by_override = 0

    def _plausible(self, fid: str, weight_class: str, fight_date: str) -> bool:
        f = self.fighters[fid]
        if f["dob"]:
            age = (date.fromisoformat(fight_date) - date.fromisoformat(f["dob"])).days / 365.25
            if not 17 <= age <= 50:
                return False
        limit = P.division_limit_lbs(weight_class)
        if f["weight"] and limit and abs(f["weight"] - limit) > 15:
            return False
        return True

    def resolve(self, name: str, weight_class: str, fight_date: str) -> str | None:
        cands = set(self.names.get(name, set()))
        pick = self.overrides.get(name)
        if pick in self.fighters and (len(cands) != 1 or pick in cands):
            self.by_override += 1
            return pick
        if len(cands) <= 1:
            return next(iter(cands), None)
        ok = [c for c in sorted(cands) if self._plausible(c, weight_class, fight_date)]
        if len(ok) == 1:
            self.by_rule += 1
            return ok[0]
        return None


def build_tables(
    snapshot: Path, overrides: dict[str, str], dropped: dict[str, str] | None = None
) -> tuple[dict, BuildReport]:
    dropped = dropped or {}
    rep = BuildReport()
    ev_rows = _read(snapshot / "ufc_event_details.csv")
    res_rows = _read(snapshot / "ufc_fight_results.csv")
    stat_rows = _read(snapshot / "ufc_fight_stats.csv")
    tott = _read(snapshot / "ufc_fighter_tott.csv")
    det = {P.fighter_id(r["URL"]): r for r in _read(snapshot / "ufc_fighter_details.csv")}

    # events
    events = {}
    for r in ev_rows:
        events[r["EVENT"].strip()] = {
            "event_id": P.fighter_id(r["URL"]),
            "name": r["EVENT"].strip(),
            "date": P.parse_date(r["DATE"]),
            "location": P.clean(r["LOCATION"]),
        }

    # fighters; every name a fighter goes by (tott name, details name) maps to their id
    fighters, names = {}, collections.defaultdict(set)
    for r in tott:
        fid = P.fighter_id(r["URL"])
        d = det.get(fid)
        detail_name = f"{d['FIRST']} {d['LAST']}".strip() if d else ""
        name = detail_name or r["FIGHTER"].strip()
        fighters[fid] = {
            "fighter_id": fid,
            "name": name,
            "height_cm": P.parse_height_cm(r["HEIGHT"]),
            "reach_cm": P.parse_reach_cm(r["REACH"]),
            "stance": P.clean(r["STANCE"]),
            "dob": P.parse_date(r["DOB"]),
            "weight": P.parse_weight_lbs(r["WEIGHT"]),
        }
        for n in {r["FIGHTER"].strip(), detail_name}:
            if n:
                names[n].add(fid)
    index = NameIndex(fighters, names, overrides)

    # fights: drop duplicates and fights whose event is unknown, with reasons
    rep.source_fights = len(res_rows)
    by_url = collections.defaultdict(list)
    for pos, r in enumerate(res_rows):
        by_url[r["URL"].strip()].append((pos, r))
    kept, key_to_url = [], collections.defaultdict(set)
    dup_dropped = unknown_event = 0
    for url, group in by_url.items():
        for _, r in group:
            key_to_url[(r["EVENT"].strip(), r["BOUT"].strip())].add(P.fighter_id(url))
        known = [(p, r) for p, r in group if r["EVENT"].strip() in events]
        if not known:
            unknown_event += len(group)
            continue
        dup_dropped += len(group) - 1
        kept.append(known[0])
    if dup_dropped:
        rep.dropped.append((dup_dropped, "same fight URL listed again under a renamed event name"))
    if unknown_event:
        rep.dropped.append((unknown_event, "event not in the events list (no date)"))
    kept.sort(key=lambda x: x[0])

    fights, order, by_reason = [], collections.Counter(), collections.Counter()
    for _, r in kept:
        event = events[r["EVENT"].strip()]
        if P.fighter_id(r["URL"]) in dropped:
            by_reason[dropped[P.fighter_id(r["URL"])]] += 1
            continue
        wc = P.parse_weight_class(r["WEIGHTCLASS"])
        n1, n2 = (x.strip() for x in r["BOUT"].split(" vs. "))
        ids = [index.resolve(n, wc["weight_class"], event["date"]) for n in (n1, n2)]
        if None in ids or ids[0] == ids[1]:
            rep.unresolved.append(f"{event['name']}: {r['BOUT'].strip()} ({r['URL'].strip()})")
            continue
        outcome = P.parse_outcome(r["OUTCOME"])
        winner = {"first_won": ids[0], "second_won": ids[1]}.get(outcome)
        if ids[0] > ids[1]:  # principle 3: id order never encodes the result
            ids.reverse()
            n1, n2 = n2, n1
        order[event["event_id"]] += 1
        fights.append(
            {
                "fight_id": P.fighter_id(r["URL"]),
                "event_id": event["event_id"],
                "date": event["date"],
                "bout_order": order[event["event_id"]],  # 1 = main event (source order)
                "fighter_1_id": ids[0],
                "fighter_2_id": ids[1],
                "winner_id": winner,
                "outcome": "win" if winner else outcome,
                "method": P.clean(r["METHOD"]),
                "end_round": P.to_int(r["ROUND"]),
                "end_time_s": P.parse_clock(r["TIME"]),  # seconds into end_round
                "scheduled_rounds": P.parse_scheduled_rounds(r["TIME FORMAT"]),
                **wc,
                "referee": P.clean(r["REFEREE"]),
                "_names": {ids[0]: n1, ids[1]: n2},
            }
        )
    rep.dropped += [(n, reason) for reason, n in by_reason.items()]
    fight_by_id = {f["fight_id"]: f for f in fights}

    # round stats: join on (event, bout); skip keys that name more than one fight
    stats, seen = [], set()
    for r in stat_rows:
        if not r["FIGHTER"].strip():
            continue
        urls = key_to_url.get((r["EVENT"].strip(), r["BOUT"].strip()), set())
        if len(urls) != 1:
            continue
        f = fight_by_id.get(next(iter(urls)))
        if f is None:
            continue
        fid = next((i for i, n in f["_names"].items() if n == r["FIGHTER"].strip()), None)
        if fid is None:
            raise ValidationError(f"stats fighter {r['FIGHTER']!r} not in bout {r['BOUT']!r}")
        rnd = P.parse_round(r["ROUND"])
        if (f["fight_id"], fid, rnd) in seen:
            continue
        seen.add((f["fight_id"], fid, rnd))
        row = {
            "fight_id": f["fight_id"],
            "fighter_id": fid,
            "round": rnd,
            "knockdowns": P.to_int(r["KD"]),
            "sub_attempts": P.to_int(r["SUB.ATT"]),
            "reversals": P.to_int(r["REV."]),
            "control_s": P.parse_clock(r["CTRL"]),
        }
        for key, col in STAT_PAIRS.items():
            row[f"{key}_landed"], row[f"{key}_att"] = P.parse_of(r[col])
        stats.append(row)
    # Keys naming the same bout twice in one event cannot be joined to stats: counted, not guessed.
    rep.ambiguous_stats_fights = sum(1 for u in key_to_url.values() if len(u) > 1)

    for f in fights:
        del f["_names"]
    fighter_rows = [
        {k: v for k, v in fighters[i].items() if k != "weight"} for i in sorted(fighters)
    ]
    event_rows = sorted(events.values(), key=lambda e: (e["date"], e["event_id"]))
    int_cols = {"end_round", "end_time_s", "scheduled_rounds", "bout_order"}
    tables = {
        "events": pd.DataFrame(event_rows),
        "fights": pd.DataFrame(fights).astype({c: "Int64" for c in int_cols}),
        "fight_stats_round": pd.DataFrame(stats),
        "fighters": pd.DataFrame(fighter_rows),
    }
    st = tables["fight_stats_round"]
    tables["fight_stats_round"] = st.astype(
        {c: "Int64" for c in st.columns if c not in ("fight_id", "fighter_id")}
    )
    tables["fights"] = (
        tables["fights"].sort_values(["date", "event_id", "bout_order"]).reset_index(drop=True)
    )
    rep.counts = {k: len(v) for k, v in tables.items()}
    rep.fights_without_round_stats = int(
        (~tables["fights"].fight_id.isin(tables["fight_stats_round"].fight_id)).sum()
    )
    rep.resolved_by_rule, rep.resolved_by_override = index.by_rule, index.by_override
    return tables, rep


def build(snapshot: Path, processed_dir: Path, overrides_dir: Path | None) -> BuildReport:
    ov = overrides_dir
    tables, rep = build_tables(
        snapshot,
        load_overrides(ov and ov / "fighter_ids.csv"),
        load_dropped(ov and ov / "dropped_fights.csv"),
    )
    if rep.unresolved:
        raise ValidationError(
            f"{len(rep.unresolved)} fights have fighters that cannot be resolved; add them to "
            "overrides/fighter_ids.csv (or overrides/dropped_fights.csv):\n- "
            + "\n- ".join(rep.unresolved)
        )
    expected = rep.source_fights - _dropped_total(rep)
    validate(tables, expected_fights=expected)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_parquet(processed_dir / f"{name}.parquet", index=False)
    return rep


def _dropped_total(rep: BuildReport) -> int:
    return sum(n for n, _ in rep.dropped)
