"""`ufc` command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from . import fetch as fetch_mod
from .backtest import LeakSuspected, run_backtest
from .backtest.ablation import ablation_markdown, run_ablation
from .backtest.protocol import DEV_END, FINAL_FEATURE_SET, HOLDOUT_START, ProtocolError
from .backtest.ratings_report import ratings_report
from .backtest.run import default_out_dir, scored_fights
from .build import build
from .features import (
    PhysicalNullError,
    UnknownFeatureSet,
    build_features,
    check_physical,
    check_truncation,
)
from .features.params import RATING_PARAMS, TUNE_CUTOFF
from .features.tune import TuneCutoffError, tune_ratings
from .odds import OddsError, build_odds, validate_odds
from .validate import ValidationError, load_tables, validate


def _build_odds(raw: Path, processed: Path, overrides: Path) -> None:
    snap = fetch_mod.latest_odds_snapshot(raw)
    if snap is None:
        (processed / "odds.parquet").unlink(missing_ok=True)  # never leave a stale table
        print("odds: no odds snapshot (run `ufc fetch`); odds.parquet not built")
        return
    odds, rep = build_odds(snap, load_tables(processed), overrides / "odds_names.csv")
    odds.to_parquet(processed / "odds.parquet", index=False)
    print(f"odds snapshot {snap.name}")
    print(
        f"odds source rows: {rep.source_total}; without usable odds: {rep.source_without_odds}; "
        f"matched fallback pass: {rep.matched_by_fallback}"
    )
    print(f"odds: {len(odds)} rows")
    print("year | source rows | with odds | matched | dropped (overround) | unmatched")
    for y, c in rep.per_year.items():
        print(
            f"{y} | {c['source']} | {c['with_odds']} | {c['matched']} | {c['dropped']} | {c['unmatched']}"
        )
    print(
        f"scored fights (decided, from 2015-01-01) with odds: "
        f"{rep.scored_with_odds} of {rep.scored_fights} "
        f"({rep.scored_with_odds / max(rep.scored_fights, 1):.1%})"
    )
    for title, lines in (
        ("unmatched odds rows", rep.unmatched),
        ("ambiguous odds rows (not matched)", rep.ambiguous),
        ("dropped for overround outside range", rep.dropped_overround),
    ):
        print(f"{title}: {len(lines)}")
        for line in lines:
            print(f"  {line}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ufc")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--overrides", type=Path, default=Path("overrides"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the source CSVs")
    f.add_argument("--sha", help="upstream commit to fetch (default: latest on main)")
    f.add_argument("--odds-sha", help="odds dataset commit to fetch (default: latest on main)")
    sub.add_parser("build", help="build and validate the Parquet tables")
    sub.add_parser("validate", help="validate the built tables")
    ft = sub.add_parser("features", help="build point-in-time features.parquet")
    ft.add_argument(
        "--check",
        action="store_true",
        help="also verify truncation invariance on a random sample of fights",
    )
    ft.add_argument("--sample", type=int, default=200, help="fights to check with --check")
    ft.add_argument("--seed", type=int, default=0, help="sampling seed for --check")
    tr = sub.add_parser(
        "tune-ratings",
        help="tune the rating constants on fights before a cutoff (default 2015-01-01)",
    )
    tr.add_argument("--before", default=TUNE_CUTOFF, help="use only fights dated before this day")
    rr = sub.add_parser(
        "ratings-report",
        help="write docs/results/ratings.md for the frozen configuration (runs the holdout once)",
    )
    rr.add_argument(
        "--frozen-commit", required=True, help="the commit that froze the configuration"
    )
    rr.add_argument("--reading", type=Path, help="hand-written text to append (marked as such)")
    rr.add_argument("--out", type=Path, default=Path("docs/results/ratings.md"))
    bt = sub.add_parser("backtest", help="walk-forward backtest and report")
    bt.add_argument("--start", default="2015-01-01", help="first day of the test period")
    bt.add_argument(
        "--end", help="last day of the test period (development ends 2022-12-31, see design.md)"
    )
    bt.add_argument(
        "--features-set",
        default=None,
        help="feature set: base, ratings, adjusted, context, all, or a join such as "
        "ratings+adjusted (default: the frozen configuration, else base)",
    )
    bt.add_argument(
        "--ablation",
        action="store_true",
        help="run every feature set on the development period and apply the decision rule",
    )
    bt.add_argument("--reports-dir", type=Path, default=Path("reports"))
    args = ap.parse_args(argv)
    raw, processed = args.data_dir / "raw", args.data_dir / "processed"

    try:
        if args.cmd == "fetch":
            snap = fetch_mod.fetch(raw, args.sha)
            print(f"snapshot {snap.name} in {snap}")
            osnap = fetch_mod.fetch_odds(raw, args.odds_sha)
            print(f"odds snapshot {osnap.name} in {osnap}")
        elif args.cmd == "build":
            snap = fetch_mod.latest_snapshot(raw)
            rep = build(snap, processed, args.overrides)
            print(f"built from snapshot {snap.name}")
            print(f"source fights: {rep.source_fights}")
            for n, reason in rep.dropped:
                print(f"dropped: {n} fights: {reason}")
            for table, n in rep.counts.items():
                print(f"{table}: {n} rows")
            print(f"fights without round stats: {rep.fights_without_round_stats}")
            print(
                f"bouts named twice in one event (stats not joined): {rep.ambiguous_stats_fights}"
            )
            print(
                f"duplicate names resolved: {rep.resolved_by_rule} by weight/age rule, "
                f"{rep.resolved_by_override} by overrides file"
            )
            print("unresolved fighter names: 0")
            print("validation passed")
            _build_odds(raw, processed, args.overrides)
        elif args.cmd == "features":
            t = load_tables(processed)
            feats = build_features(t["fights"], t["fight_stats_round"], t["fighters"])
            check_physical(feats)
            feats.to_parquet(processed / "features.parquet", index=False)
            print(f"features: {len(feats)} rows, {feats.shape[1]} columns")
            if args.check:
                bad = check_truncation(
                    t["fights"],
                    t["fight_stats_round"],
                    t["fighters"],
                    feats,
                    args.sample,
                    args.seed,
                )
                if bad:
                    print(f"LEAK: truncation check failed for fights {bad}", file=sys.stderr)
                    return 1
                n = min(args.sample, len(feats))
                print(f"truncation invariance holds on {n} sampled fights")
        elif args.cmd == "ratings-report":
            t = load_tables(processed)
            feats = pd.read_parquet(processed / "features.parquet")
            check_physical(feats)
            odds_path = processed / "odds.parquet"
            if not odds_path.exists():
                raise FileNotFoundError("no odds.parquet: run `ufc fetch` and `ufc build` first")
            osnap = fetch_mod.latest_odds_snapshot(raw)
            text = ratings_report(
                feats,
                t["fights"],
                t["fight_stats_round"],
                pd.read_parquet(odds_path),
                snapshot=fetch_mod.latest_snapshot(raw).name,
                odds_snapshot=osnap.name if osnap else "unknown",
                frozen_commit=args.frozen_commit,
                reading=args.reading.read_text() if args.reading else None,
            )
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(text)
            print(f"wrote {args.out}")
        elif args.cmd == "tune-ratings":
            t = load_tables(processed)
            params, diag = tune_ratings(t["fights"], t["fight_stats_round"], args.before)
            print(f"tuned on {diag['fights']} fights dated before {args.before}; one-step-ahead")
            print(f"error scored from {diag['scored_from']}")
            g, w, a = diag["glicko"], diag["welo"], diag["adjusted"]
            print(f"glicko: c={g['c']} log loss {g['log_loss']:.4f} (c=0: {g['log_loss_c0']:.4f})")
            print(
                f"method-weighted Elo: log loss {w['log_loss']:.4f} (plain Elo K=32 {w['log_loss_plain_elo']:.4f})"
            )
            print(
                f"adjusted: k0={a['k0']} rate_min={a['rate_min']} standardised mse {a['mse']:.4f} "
                f"(k0=5, rate_min=0.05: {a['mse_untuned']:.4f})"
            )
            print(params)
            if args.before != TUNE_CUTOFF:
                print(
                    f"not compared with the committed constants (they use --before {TUNE_CUTOFF})"
                )
            elif params == RATING_PARAMS:
                print("matches the committed constants")
            else:
                print(f"DIFFERS from the committed constants: {RATING_PARAMS}", file=sys.stderr)
                return 1
        elif args.cmd == "backtest":
            t = load_tables(processed)
            feats = pd.read_parquet(processed / "features.parquet")
            check_physical(feats)  # a stale pre-fix features file must not run silently
            try:
                snap = fetch_mod.latest_snapshot(raw).name
            except FileNotFoundError:
                snap = "unknown"
            out = default_out_dir(args.reports_dir, snap)
            fset = args.features_set or FINAL_FEATURE_SET or "base"
            if (args.end is None or args.end > DEV_END) and fset != "base" and not args.ablation:
                print(
                    f"note: this run includes the holdout ({HOLDOUT_START} on) for the frozen "
                    "configuration; the protocol reports it once (`ufc ratings-report`)",
                    file=sys.stderr,
                )
            cmd = "uv run ufc backtest" + (
                f" --start {args.start}" if args.start != "2015-01-01" else ""
            )
            cmd += f" --end {args.end}" if args.end else ""
            cmd += f" --features-set {args.features_set}" if args.features_set else ""
            if args.ablation:
                if args.end or args.features_set or args.start != "2015-01-01":
                    raise ProtocolError(
                        "--ablation runs every feature set on the fixed development period; "
                        "do not combine it with --start, --end or --features-set"
                    )
                scored = scored_fights(feats, t["fights"])
                preds = run_ablation(scored)
                md, kept = ablation_markdown(preds)
                out.mkdir(parents=True, exist_ok=True)
                (out / "ablation.md").write_text(md + "\n")
                print(md)
                print(f"report in {out}")
                return 0
            odds_path = processed / "odds.parquet"
            odds = pd.read_parquet(odds_path) if odds_path.exists() else None
            osnap = fetch_mod.latest_odds_snapshot(raw)
            if odds is None:
                print("no odds.parquet (run `ufc build` after `ufc fetch`): market not scored")
            try:
                run_backtest(
                    feats,
                    t["fights"],
                    args.start,
                    out,
                    snap,
                    cmd,
                    odds,
                    osnap.name if osnap else "unknown",
                    fset,
                    args.end,
                )
            except LeakSuspected as e:
                print(f"LEAK LIKELY: {e} (report written to {out})", file=sys.stderr)
                return 1
            print((out / "report.md").read_text())
            print(f"report in {out}")
        else:
            rep = validate(load_tables(processed))
            print(
                f"validation passed; fights without round stats: {rep['fights_without_round_stats']}"
            )
            if (processed / "odds.parquet").exists():
                validate_odds(
                    pd.read_parquet(processed / "odds.parquet"), load_tables(processed)["fights"]
                )
                print("odds validation passed")
    except (
        ValidationError,
        FileNotFoundError,
        PhysicalNullError,
        OddsError,
        ProtocolError,
        TuneCutoffError,
        UnknownFeatureSet,
    ) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
