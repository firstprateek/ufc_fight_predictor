"""`ufc` command line."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import fetch as fetch_mod
from .build import build
from .features import build_features, check_truncation
from .validate import ValidationError, load_tables, validate


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ufc")
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--overrides", type=Path, default=Path("overrides"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch", help="download the source CSVs")
    f.add_argument("--sha", help="upstream commit to fetch (default: latest on main)")
    sub.add_parser("build", help="build and validate the Parquet tables")
    sub.add_parser("validate", help="validate the built tables")
    ft = sub.add_parser("features", help="build point-in-time features.parquet")
    ft.add_argument(
        "--check",
        action="store_true",
        help="also verify truncation invariance on a random sample of fights",
    )
    ft.add_argument("--sample", type=int, default=200, help="fights to check with --check")
    args = ap.parse_args(argv)
    raw, processed = args.data_dir / "raw", args.data_dir / "processed"

    try:
        if args.cmd == "fetch":
            snap = fetch_mod.fetch(raw, args.sha)
            print(f"snapshot {snap.name} in {snap}")
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
        elif args.cmd == "features":
            t = load_tables(processed)
            feats = build_features(t["fights"], t["fight_stats_round"], t["fighters"])
            feats.to_parquet(processed / "features.parquet", index=False)
            print(f"features: {len(feats)} rows, {feats.shape[1]} columns")
            if args.check:
                bad = check_truncation(
                    t["fights"], t["fight_stats_round"], t["fighters"], feats, args.sample
                )
                if bad:
                    print(f"LEAK: truncation check failed for fights {bad}", file=sys.stderr)
                    return 1
                n = min(args.sample, len(feats))
                print(f"truncation invariance holds on {n} sampled fights")
        else:
            rep = validate(load_tables(processed))
            print(
                f"validation passed; fights without round stats: {rep['fights_without_round_stats']}"
            )
    except (ValidationError, FileNotFoundError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
