"""Validation of the built tables. Fails loudly; reports soft findings."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

TABLES = ["events", "fights", "fight_stats_round", "fighters"]


class ValidationError(Exception):
    pass


def load_tables(processed_dir: Path) -> dict[str, pd.DataFrame]:
    return {t: pd.read_parquet(processed_dir / f"{t}.parquet") for t in TABLES}


def validate(tables: dict[str, pd.DataFrame], expected_fights: int | None = None) -> dict:
    """Raise ValidationError listing every failed check; return soft-finding counts."""
    ev, fi, st, fr = (tables[t] for t in TABLES)
    errors: list[str] = []

    def check(ok: bool, msg: str) -> None:
        if not ok:
            errors.append(msg)

    check(ev.event_id.is_unique, "events: event_id not unique")
    check(fi.fight_id.is_unique, "fights: fight_id not unique")
    check(fr.fighter_id.is_unique, "fighters: fighter_id not unique")
    check(not st.duplicated(["fight_id", "fighter_id", "round"]).any(), "stats: duplicate key")

    check(fi.event_id.isin(ev.event_id).all(), "fights: event_id missing from events")
    for col in ("fighter_1_id", "fighter_2_id"):
        check(fi[col].notna().all(), f"fights: {col} has nulls")
        check(fi[col].isin(fr.fighter_id).all(), f"fights: {col} missing from fighters")
    check((fi.fighter_1_id != fi.fighter_2_id).all(), "fights: a fighter fights themself")
    check((fi.fighter_1_id < fi.fighter_2_id).all(), "fights: fighter_1_id < fighter_2_id broken")

    win = fi.outcome == "win"
    check(
        fi.winner_id[win].notna().all()
        and ((fi.winner_id == fi.fighter_1_id) | (fi.winner_id == fi.fighter_2_id))[win].all(),
        "fights: winner_id is not one of the two fighters",
    )
    check(fi.winner_id[~win].isna().all(), "fights: winner_id set for a draw / no contest")
    check(fi.outcome.isin(["win", "draw", "nc"]).all(), "fights: bad outcome value")

    for name, col in (("events.date", ev.date), ("fights.date", fi.date), ("fighters.dob", fr.dob)):
        parsed = pd.to_datetime(col, format="%Y-%m-%d", errors="coerce")
        check(not (parsed.isna() & col.notna()).any(), f"{name}: unparseable date")
    check(ev.date.notna().all() and fi.date.notna().all(), "dates: null event/fight date")

    for col in st.columns:
        if col.endswith("_landed"):
            att = st[col[: -len("landed")] + "att"]
            bad = (st[col] > att).fillna(False)
            check(not bad.any(), f"stats: {col} > attempted in {int(bad.sum())} rows")

    # Round stats: a fight with any stats has both fighters for exactly rounds 1..end_round.
    have = st.groupby("fight_id").agg(
        rounds=("round", "nunique"), last=("round", "max"), fighters=("fighter_id", "nunique")
    )
    have = have.join(fi.set_index("fight_id")[["end_round"]], how="left")
    check(have.end_round.notna().all(), "stats: fight_id missing from fights")
    incomplete = (
        (have.fighters != 2) | (have.rounds != have.end_round) | (have["last"] != have.end_round)
    )
    check(
        not incomplete.any(), f"stats: {int(incomplete.sum())} fights have incomplete round stats"
    )
    per_fighter = (
        st.groupby(["fight_id", "fighter_id"])["round"].count().groupby("fight_id").nunique()
    )
    check(not (per_fighter > 1).any(), "stats: fighters in a fight have different round counts")

    if expected_fights is not None:
        check(len(fi) == expected_fights, f"fights: {len(fi)} rows, expected {expected_fights}")

    if errors:
        raise ValidationError("validation failed:\n- " + "\n- ".join(errors))
    return {"fights_without_round_stats": int((~fi.fight_id.isin(st.fight_id)).sum())}
