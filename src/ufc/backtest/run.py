"""Walk-forward backtest and report."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import swap_orientation
from . import metrics as M
from .models import MODELS, LeakSuspected, flagged

TRIPWIRE_ACCURACY = 0.75
BOOTSTRAP_DRAWS = 1000


def scored_fights(features: pd.DataFrame, fights: pd.DataFrame) -> pd.DataFrame:
    """Features joined to the label, for outcome == 'win' fights only. y = 1 if fighter 1 won."""
    f = fights.loc[fights.outcome == "win", ["fight_id", "winner_id"]]
    df = features.merge(f, on="fight_id", how="inner")
    df["y"] = (df.winner_id == df.fighter_1_id).astype(int)
    return df.drop(columns="winner_id").sort_values(["date", "fight_id"]).reset_index(drop=True)


def _quarter_start(d: str) -> str:
    q = (int(d[5:7]) - 1) // 3 * 3 + 1
    return f"{d[:4]}-{q:02d}-01"


def walk_forward(scored: pd.DataFrame, start: str) -> pd.DataFrame:
    """For each calendar quarter from `start`, fit every model on scored fights dated before the
    quarter starts and predict the quarter. Returns one row per test fight with a p_<model> column."""
    qstart = scored.date.map(_quarter_start)
    out = []
    for q in sorted(set(qstart[scored.date >= start])):
        train = scored[scored.date < q]
        test = scored[(qstart == q) & (scored.date >= start)]
        assert train.date.max() < q <= test.date.min(), "training data must precede the quarter"
        part = test[["fight_id", "event_id", "date", "y"]].copy()
        part["quarter"] = q
        part["snapshot_imputed"] = flagged(test)
        sw = swap_orientation(test)
        if not swap_orientation(sw).equals(test):
            raise LeakSuspected("swap_orientation is not an involution")
        for name, model in MODELS.items():
            p = model(train, train.y.to_numpy(), test)
            part[f"p_{name}"] = p
            if name in ("logreg", "hgb"):  # orientation guard: swapping the fighters gives 1 - p
                p_swapped = model(train, train.y.to_numpy(), sw)
                if not np.allclose(p + p_swapped, 1.0, atol=1e-9):
                    raise LeakSuspected(f"{name}: swapping fighters did not give 1 - p")
        out.append(part)
    if not out:
        raise ValueError(f"no scored fights on or after {start}")
    return pd.concat(out, ignore_index=True)


def _table(rows: list[list], header: list[str]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def _fmt(s: dict) -> list:
    return [
        s["n"],
        f"{s['log_loss']:.4f}",
        f"{s['brier']:.4f}",
        f"{s['accuracy']:.3f}",
        f"{s['ece']:.4f}",
    ]


HEADER = ["model", "fights", "log loss", "Brier", "accuracy", "ECE"]


def summary_markdown(pred: pd.DataFrame) -> tuple[str, dict]:
    """The overall table, bootstrap intervals and calibration of the best model."""
    names = list(MODELS)
    overall = {m: M.summarize(pred[f"p_{m}"], pred.y) for m in names}
    best = min(names, key=lambda m: overall[m]["log_loss"])
    md = ["### Overall", "", _table([[m, *_fmt(overall[m])] for m in names], HEADER), ""]
    ev = pred.event_id.to_numpy()
    loss = {m: M.point_losses(pred[f"p_{m}"].to_numpy(), pred.y.to_numpy()) for m in names}
    rows = []
    for m in names:
        if m == "elo":
            continue
        d, lo, hi = M.paired_bootstrap(loss[m], loss["elo"], ev, BOOTSTRAP_DRAWS)
        rows.append([m, f"{d:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"])
    md += [
        f"### Log loss minus Elo (paired bootstrap over events, {BOOTSTRAP_DRAWS} draws, 95%)",
        "",
        "Negative means better than Elo.",
        "",
        _table(rows, ["model", "difference", "95% interval"]),
        "",
        f"### Calibration of the best model ({best}, both fighter views)",
        "",
    ]
    cal = M.calibration_table(pred[f"p_{best}"], pred.y)
    md.append(
        _table(
            [[f"{lo:.1f}-{hi:.1f}", n, f"{mp:.3f}", f"{obs:.3f}"] for lo, hi, n, mp, obs in cal],
            ["p bin", "fighter views", "mean predicted", "observed win rate"],
        )
    )
    return "\n".join(md), overall


def flag_breakdown(pred: pd.DataFrame) -> str:
    """Scored fights with a filled-in snapshot value versus without (never trained on)."""
    rows = []
    for m in ("logreg", "hgb"):
        for label, g in (
            ("flagged", pred[pred.snapshot_imputed]),
            ("unflagged", pred[~pred.snapshot_imputed]),
        ):
            if len(g):
                s = M.summarize(g[f"p_{m}"], g.y)
                rows.append([m, label, s["n"], f"{s['log_loss']:.4f}", f"{s['accuracy']:.3f}"])
    return _table(rows, ["model", "subset", "fights", "log loss", "accuracy"])


def _train_lines(pred, info) -> list[str]:
    n_flag = int(pred.snapshot_imputed.sum())
    lines = [
        "- models train only on fights where neither fighter has a filled-in snapshot value "
        "(height, reach, stance, dob); `snapshot_imputed` flags are never model inputs",
        f"- scored fights in the test period carrying a flag: {n_flag} of {len(pred)}",
    ]
    if info:
        lines += [
            f"- decided fights excluded from training, all history: {info['excluded_all']} "
            f"of {info['scored_all']}",
            f"- last training fold ({info['last_quarter']}): {info['excluded_last']} of "
            f"{info['train_last']} fights excluded",
        ]
    return lines


def report_markdown(
    pred, start, snapshot, command, features_rows, train_info=None
) -> tuple[str, dict]:
    summary, overall = summary_markdown(pred)
    pred = pred.assign(year=pred.date.str[:4])
    rows = []
    for y, g in pred.groupby("year"):
        for m in MODELS:
            rows.append([y, m, *_fmt(M.summarize(g[f"p_{m}"], g.y))])
    head = [
        "# Baseline backtest",
        "",
        f"- snapshot: `{snapshot}`",
        f"- command: `{command}`",
        f"- test period: {pred.date.min()} to {pred.date.max()} (start {start}), "
        f"{len(pred)} scored fights (win outcomes only), {pred.event_id.nunique()} events",
        f"- features rows: {features_rows}",
        "- each quarter: fit on all scored fights dated before the quarter, predict the quarter",
        "- ECE: 10 equal-width bins, each fight counted from both fighters' views",
        "- accuracy: p = 0.5 counts as half a hit",
        *_train_lines(pred, train_info),
        "",
        summary,
        "",
        "### Per year",
        "",
        _table(rows, ["year", *HEADER]),
        "",
        "### Scored fights with a filled-in snapshot value (flagged) versus without",
        "",
        "Models never train on flagged fights; they are still predicted and scored.",
        "",
        flag_breakdown(pred),
        "",
    ]
    return "\n".join(head), overall


def run_backtest(
    features: pd.DataFrame,
    fights: pd.DataFrame,
    start: str,
    out_dir: Path | None,
    snapshot: str = "unknown",
    command: str = "uv run ufc backtest",
) -> dict:
    """Run the walk-forward backtest, write the report, and enforce the leak tripwire."""
    scored = scored_fights(features, fights)
    pred = walk_forward(scored, start)
    last_q = pred.quarter.max()
    train_last = scored[scored.date < last_q]
    info = {
        "scored_all": len(scored),
        "excluded_all": int(flagged(scored).sum()),
        "last_quarter": last_q,
        "train_last": len(train_last),
        "excluded_last": int(flagged(train_last).sum()),
    }
    md, overall = report_markdown(pred, start, snapshot, command, len(features), info)
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "report.md").write_text(md + "\n")
        pred.to_parquet(out_dir / "predictions.parquet", index=False)
    too_good = {m: s["accuracy"] for m, s in overall.items() if s["accuracy"] > TRIPWIRE_ACCURACY}
    if too_good:
        raise LeakSuspected(
            f"accuracy above {TRIPWIRE_ACCURACY:.0%} for {too_good}: a leak is likely "
            "(real UFC accuracy is roughly 55-65%)"
        )
    return overall


def default_out_dir(root: Path, snapshot: str) -> Path:
    return root / f"{snapshot}-{date.today().isoformat()}"
