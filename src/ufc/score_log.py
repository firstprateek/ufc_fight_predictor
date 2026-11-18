"""`ufc score-log`: score the prediction log against what happened afterwards.

The 2023 to 2026 holdout has been used, so the only fair test from now on is a prediction written
down before a fight and scored after it. The rules that make it hard to flatter the model by accident:

* a row is matched to the earliest fight of the same two fighters dated on or after the day it was
  written and at most `MATCH_WINDOW_DAYS` after the fight date it names;
* a row counts only when the UTC day it was written is strictly before the fight's date;
* one prediction per fight: the latest eligible row, never chosen by probability or outcome.

The log is only ever read.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import metrics
from .backtest.run import BOOTSTRAP_DRAWS
from .odds import OVERROUND_RANGE, remove_margin
from .predict import LOG_COLUMNS, PredictError

MATCH_WINDOW_DAYS = (
    45  # a fight counts as the logged one up to this many days after its logged date
)
MIN_SCORED_FIGHTS = 200  # below this many scored fights the result cannot separate the models
COIN_FLIP = {"log_loss": float(np.log(2)), "brier": 0.25, "accuracy": 0.5}
MODELS = {"blend": "p_blend_1", "hgb": "p_hgb_1", "logreg": "p_logreg_1"}
MODEL_LABELS = {"blend": "blend", "hgb": "hgb", "logreg": "logistic regression"}

SCORED = "scored"
SUPERSEDED = "superseded"
TOO_LATE = "logged too late"
PENDING = "pending"
DID_NOT_HAPPEN = "did not happen"
NO_DECISION = "no decision"
STATUSES = [SCORED, SUPERSEDED, TOO_LATE, PENDING, DID_NOT_HAPPEN, NO_DECISION]

BACKTEST_NOTE = (
    "The backtest result (market 0.6092 against 0.6474 for the model over 5,179 fights) "
    "remains the best evidence."
)
RULE_NOTE = (
    "Only predictions written before the fight count: a row whose run date (UTC) is not strictly "
    "before the fight's date is reported as logged too late and never scored. Log at least the "
    "day before."
)


class ScoreLogError(PredictError):
    pass


@dataclass
class Row:
    line: int
    run: datetime
    day: date  # UTC date of run_utc
    fight_date: date  # the `date` column
    f1: str
    n1: str
    f2: str
    n2: str
    probs: dict[str, float]
    odds: tuple[float, float] | None
    commit: str
    snapshot: str


# ---------------------------------------------------------------- reading


def _parse_day(s: str, where: str) -> date:
    try:
        return date.fromisoformat(s.strip())
    except ValueError:
        raise ScoreLogError(f"{where}: cannot read date {s!r} (want YYYY-MM-DD)") from None


def _parse_run(s: str, where: str) -> datetime:
    try:
        t = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    except ValueError:
        raise ScoreLogError(
            f"{where}: cannot read run_utc {s!r} (want YYYY-MM-DDTHH:MM:SSZ)"
        ) from None
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _number(s: str, where: str, what: str) -> float:
    try:
        x = float(s)
    except ValueError:
        raise ScoreLogError(f"{where}: {what} {s!r} is not a number") from None
    if not np.isfinite(x):
        raise ScoreLogError(f"{where}: {what} {s!r} is not a finite number")
    return x


def read_log(path: Path, fighter_ids: set[str]) -> list[Row]:
    """The rows of the log. Any malformed row fails naming its line number; none is skipped."""
    rows: list[Row] = []
    with Path(path).open(newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, [])
        if header != LOG_COLUMNS:
            raise ScoreLogError(
                f"{path}: the header is not the prediction log's. Expected: {','.join(LOG_COLUMNS)}"
            )
        for rec in reader:
            line = reader.line_num
            where = f"{path} line {line}"
            if not rec:
                continue  # a blank line carries no row
            if len(rec) != len(LOG_COLUMNS):
                raise ScoreLogError(f"{where}: {len(rec)} fields, expected {len(LOG_COLUMNS)}")
            r = dict(zip(LOG_COLUMNS, rec, strict=True))
            run = _parse_run(r["run_utc"], where)
            fight_date = _parse_day(r["date"], where)
            for k in ("fighter_1_id", "fighter_2_id"):
                if r[k] not in fighter_ids:
                    raise ScoreLogError(f"{where}: unknown fighter id {r[k]!r} in {k}")
            if r["fighter_1_id"] == r["fighter_2_id"]:
                raise ScoreLogError(f"{where}: the same fighter on both sides")
            probs = {}
            for m, col in MODELS.items():
                p = _number(r[col], where, col)
                if not 0 < p < 1:
                    raise ScoreLogError(f"{where}: {col} = {p} is outside (0, 1)")
                probs[m] = p
            odds = None
            o1, o2 = r["odds_1"].strip(), r["odds_2"].strip()
            if o1 or o2:
                if not (o1 and o2):
                    raise ScoreLogError(f"{where}: only one of odds_1 and odds_2 is given")
                a, b = _number(o1, where, "odds_1"), _number(o2, where, "odds_2")
                if abs(a) < 100 or abs(b) < 100:
                    raise ScoreLogError(f"{where}: odds {a:g} / {b:g} are not American prices")
                over = remove_margin(a, b)[2]
                lo, hi = OVERROUND_RANGE
                if not lo <= round(over, 9) <= hi:
                    raise ScoreLogError(
                        f"{where}: odds {a:g} / {b:g} imply a total of {over:.3f}, "
                        f"outside {lo} to {hi}"
                    )
                odds = (a, b)
            rows.append(
                Row(
                    line=line,
                    run=run,
                    day=run.date(),
                    fight_date=fight_date,
                    f1=r["fighter_1_id"],
                    n1=r["fighter_1_name"],
                    f2=r["fighter_2_id"],
                    n2=r["fighter_2_name"],
                    probs=probs,
                    odds=odds,
                    commit=r["code_commit"],
                    snapshot=r["data_snapshot"],
                )
            )
    return rows


# ---------------------------------------------------------------- matching


def _classify(rows: list[Row], fights: pd.DataFrame) -> list[dict]:
    """One entry per row: its status, its reason and the fight it matched (if any)."""
    by_pair: dict[frozenset, list[dict]] = {}
    for f in fights.sort_values(["date", "bout_order", "fight_id"]).to_dict("records"):
        by_pair.setdefault(frozenset((f["fighter_1_id"], f["fighter_2_id"])), []).append(
            {**f, "_d": date.fromisoformat(f["date"])}
        )
    latest = max(date.fromisoformat(d) for d in fights["date"])
    window = timedelta(days=MATCH_WINDOW_DAYS)
    out = []
    for r in rows:
        pair = by_pair.get(frozenset((r.f1, r.f2)), [])
        end = r.fight_date + window
        entry = {"row": r, "fight": None, "status": None, "reason": ""}
        hit = next((f for f in pair if r.day <= f["_d"] <= end), None)
        if hit is not None:
            entry["fight"] = hit
            if r.day >= hit["_d"]:
                entry["status"] = TOO_LATE
                entry["reason"] = f"written on the fight date ({hit['_d']}), not before it"
            else:
                entry["status"] = "eligible"
        else:
            # a fight that was already over when the row was written is never a match
            past = next(
                (f for f in pair if r.fight_date - window <= f["_d"] <= end and f["_d"] < r.day),
                None,
            )
            if past is not None:
                entry["fight"] = past
                entry["status"] = TOO_LATE
                entry["reason"] = f"written on {r.day}, after the fight on {past['_d']}"
            elif latest <= end:
                entry["status"] = PENDING
                entry["reason"] = f"no fight yet; the window runs to {end}"
            else:
                entry["status"] = DID_NOT_HAPPEN
                entry["reason"] = f"no fight between these two by {end}"
        out.append(entry)
    # one prediction per fight: the latest eligible row (run time, then file order)
    groups: dict[str, list[dict]] = {}
    for e in out:
        if e["status"] == "eligible":
            groups.setdefault(e["fight"]["fight_id"], []).append(e)
    for g in groups.values():
        g.sort(key=lambda e: (e["row"].run, e["row"].line))
        for e in g[:-1]:
            e["status"] = SUPERSEDED
            e["reason"] = f"a later row (line {g[-1]['row'].line}) is the prediction for this fight"
        last = g[-1]
        if last["fight"].get("outcome") != "win" or not isinstance(last["fight"]["winner_id"], str):
            last["status"] = NO_DECISION
            last["reason"] = (
                f"the fight on {last['fight']['_d']} ended in a {last['fight']['outcome']}"
            )
        else:
            last["status"] = SCORED
    return out


# ---------------------------------------------------------------- scoring


def _table(p: np.ndarray, y: np.ndarray) -> dict:
    return {
        "fights": int(len(y)),
        "log_loss": metrics.log_loss(p, y),
        "brier": metrics.brier(p, y),
        "accuracy": metrics.accuracy(p, y),
    }


def score(rows: list[Row], fights: pd.DataFrame) -> dict:
    entries = _classify(rows, fights)
    counts = {s: sum(e["status"] == s for e in entries) for s in STATUSES}
    scored = [e for e in entries if e["status"] == SCORED]
    scored.sort(key=lambda e: (e["fight"]["_d"], e["row"].line))
    n = len(scored)
    doc: dict = {
        "rule": RULE_NOTE,
        "latest_fight_date": str(max(fights["date"])),
        "rows": len(rows),
        "scored_fights": n,
        "min_scored_fights": MIN_SCORED_FIGHTS,
        "too_few_note": None,
        "coin_flip": COIN_FLIP,
        "models": None,
        "market": None,
        "fights": [],
        "status_counts": counts,
        "not_scored": [],
        "provenance": None,
    }
    if n < MIN_SCORED_FIGHTS:
        doc["too_few_note"] = (
            f"{n} scored fights is too few to tell the model from a coin flip or from the market. "
            + BACKTEST_NOTE
        )
    for e in entries:
        if e["status"] in (TOO_LATE, DID_NOT_HAPPEN, NO_DECISION):
            r = e["row"]
            doc["not_scored"].append(
                {
                    "line": r.line,
                    "status": e["status"],
                    "fighter_1": r.n1,
                    "fighter_2": r.n2,
                    "logged_date": str(r.fight_date),
                    "run_utc": r.run.strftime("%Y-%m-%dT%H:%M:%SZ"),
                    "reason": e["reason"],
                }
            )
    if n == 0:
        return doc

    y = np.array([float(e["fight"]["winner_id"] == e["row"].f1) for e in scored])
    p = {m: np.array([e["row"].probs[m] for e in scored]) for m in MODELS}
    doc["models"] = {m: _table(p[m], y) for m in MODELS}

    pm = np.array([remove_margin(*e["row"].odds)[0] if e["row"].odds else np.nan for e in scored])
    has = ~np.isnan(pm)
    market: dict = {"fights": int(has.sum()), "events": 0, "difference": None, "note": None}
    if has.any():
        ev = np.array([e["fight"]["event_id"] for e in scored])[has]
        market["events"] = int(len(np.unique(ev)))
        market["market"] = _table(pm[has], y[has])
        market["blend"] = _table(p["blend"][has], y[has])
        if market["events"] >= 2:
            lb = metrics.point_losses(p["blend"][has], y[has])
            lm = metrics.point_losses(pm[has], y[has])
            d, lo, hi = metrics.paired_bootstrap(lb, lm, ev, BOOTSTRAP_DRAWS)
            market["difference"] = {"mean": d, "low": lo, "high": hi, "draws": BOOTSTRAP_DRAWS}
        else:
            market["note"] = (
                "no interval: it needs at least 2 events and these fights come from "
                f"{market['events']}"
            )
    doc["market"] = market

    losses = metrics.point_losses(p["blend"], y)
    for i, e in enumerate(scored):
        r, f = e["row"], e["fight"]
        doc["fights"].append(
            {
                "line": r.line,
                "fight_date": f["date"],
                "fighter_1": r.n1,
                "fighter_2": r.n2,
                "p_blend_1": r.probs["blend"],
                "p_market_1": None if np.isnan(pm[i]) else float(pm[i]),
                "winner": r.n1 if y[i] == 1 else r.n2,
                "log_loss_blend": float(losses[i]),
            }
        )
    bad = [e for e in scored if e["row"].commit.endswith("-dirty") or e["row"].commit == "unknown"]
    doc["provenance"] = {
        "code_commits": len({e["row"].commit for e in scored}),
        "data_snapshots": len({e["row"].snapshot for e in scored}),
        "dirty_or_unknown_commit": len(bad),
    }
    return doc


# ---------------------------------------------------------------- output


def _metrics_line(label: str, t: dict) -> str:
    return f"  {label:<20}{t['fights']:>7}{t['log_loss']:>10.4f}{t['brier']:>8.4f}{t['accuracy']:>10.1%}"


def render(doc: dict) -> str:
    out = ["Score of the prediction log", ""]
    out.append(doc["rule"])
    out.append(f"Latest fight in the data: {doc['latest_fight_date']}.")
    out.append("")
    out.append(f"Scored fights: {doc['scored_fights']}")
    if doc["too_few_note"]:
        out.append(doc["too_few_note"])
    out.append("")
    head = f"  {'':<20}{'fights':>7}{'log loss':>10}{'Brier':>8}{'accuracy':>10}"
    if doc["models"]:
        out += ["Models", head]
        for m, t in doc["models"].items():
            out.append(_metrics_line(MODEL_LABELS[m], t))
        c = doc["coin_flip"]
        out.append(
            f"  {'coin flip':<20}{'':>7}{c['log_loss']:>10.4f}{c['brier']:>8.4f}{c['accuracy']:>10.1%}"
        )
        out.append("")
        mk = doc["market"]
        out.append(f"Against the market (scored fights with both prices: {mk['fights']})")
        if mk["fights"] == 0:
            out.append("  no scored fight has both prices")
        else:
            out.append(head)
            out.append(_metrics_line("market, margin out", mk["market"]))
            out.append(_metrics_line("blend, same fights", mk["blend"]))
            d = mk["difference"]
            if d:
                out.append(
                    f"  blend minus market, log loss: {d['mean']:+.4f}, 95% interval "
                    f"[{d['low']:+.4f}, {d['high']:+.4f}] "
                    f"(paired bootstrap over {mk['events']} events, {d['draws']:,} draws; "
                    "positive means the market was better)"
                )
            else:
                out.append(f"  blend minus market: {mk['note']}")
        out.append("")
        out.append("Scored fights, oldest first")
        out.append(
            "  date        fighter 1 vs fighter 2 | blend p(1) | market p(1) | winner | blend log loss"
        )
        for f in doc["fights"]:
            mp = "-" if f["p_market_1"] is None else f"{f['p_market_1']:.3f}"
            out.append(
                f"  {f['fight_date']}  {f['fighter_1']} vs {f['fighter_2']} | "
                f"{f['p_blend_1']:.3f} | {mp} | {f['winner']} | {f['log_loss_blend']:.4f}"
            )
        out.append("")
        pv = doc["provenance"]
        out.append(
            f"Provenance of the scored rows: {pv['code_commits']} code commit(s), "
            f"{pv['data_snapshots']} data snapshot(s); {pv['dirty_or_unknown_commit']} row(s) from a "
            "commit marked -dirty or unknown (kept, not excluded)."
        )
        out.append("")
    out.append("Rows by status")
    for s in STATUSES:
        out.append(f"  {s:<16}{doc['status_counts'][s]:>5}")
    out.append(f"  {'total':<16}{doc['rows']:>5}")
    if doc["not_scored"]:
        out += ["", "Rows not scored, with the reason"]
        for r in doc["not_scored"]:
            out.append(
                f"  line {r['line']}: {r['status']}: {r['fighter_1']} vs {r['fighter_2']} "
                f"(logged date {r['logged_date']}, run {r['run_utc']}): {r['reason']}"
            )
    return "\n".join(out) + "\n"


def to_json(doc: dict) -> str:
    return json.dumps(doc, indent=2)


def run(log: Path, processed: Path) -> dict | None:
    """Score the log at `log` against the built tables; None when there is no log to read."""
    log = Path(log)
    if not log.exists() or log.stat().st_size == 0:
        return None
    fights = pd.read_parquet(processed / "fights.parquet")
    ids = set(pd.read_parquet(processed / "fighters.parquet")["fighter_id"])
    return score(read_log(log, ids), fights)
