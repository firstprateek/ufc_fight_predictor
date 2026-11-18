from pathlib import Path

import pandas as pd
import pytest

from ufc.build import build, build_tables, load_dropped, load_overrides
from ufc.validate import ValidationError, load_tables, validate

FIX = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def built():
    tables, rep = build_tables(
        FIX / "source",
        load_overrides(FIX / "overrides" / "fighter_ids.csv"),
        load_dropped(FIX / "overrides" / "dropped_fights.csv"),
    )
    return tables, rep


def fight(tables, fight_id):
    f = tables["fights"]
    return f[f.fight_id == fight_id].iloc[0]


def test_counts_and_drops(built):
    tables, rep = built
    assert rep.source_fights == 9
    assert rep.dropped == [(1, "same fight URL listed again under a renamed event name")]
    assert len(tables["fights"]) == 8
    assert len(tables["events"]) == 3  # the future event is kept in events (no fights)
    assert rep.unresolved == []
    validate(tables, expected_fights=8)


def test_fighters_never_ordered_by_result(built):
    f = built[0]["fights"]
    assert (f.fighter_1_id < f.fighter_2_id).all()
    # fight 1: Alice (b...) beat Bea (a...). Winner is listed first in the source, stored second.
    r = fight(built[0], "f000000000000001")
    assert r.fighter_1_id == "a000000000000002" and r.winner_id == "b000000000000001"
    # fight 2: second-named fighter won (L/W)
    r = fight(built[0], "f000000000000002")
    assert r.winner_id == "b000000000000004" and r.outcome == "win"


def test_draw_and_no_contest(built):
    d = fight(built[0], "f000000000000005")
    assert d.outcome == "draw" and pd.isna(d.winner_id)
    n = fight(built[0], "f000000000000006")
    assert n.outcome == "nc" and pd.isna(n.winner_id)
    assert pd.isna(n.scheduled_rounds)  # 'No Time Limit'


def test_fight_fields(built):
    r = fight(built[0], "f000000000000001")
    assert (r.weight_class, r.is_title, r.gender) == ("Women's Strawweight", True, "female")
    assert (r.end_round, r.end_time_s, r.scheduled_rounds, r.method) == (2, 90, 5, "KO/TKO")
    assert r.date == "2020-01-18" and r.bout_order == 1 and r.referee == "Ref One"
    assert pd.isna(fight(built[0], "f000000000000003").referee)


def test_duplicate_names(built):
    tables, rep = built
    # Mike Davis by override, Sam Stone by weight rule (Middleweight -> the 185 lb one)
    assert fight(tables, "f000000000000003").fighter_1_id == "d000000000000001"
    assert "5000000000000002" in {
        fight(tables, "f000000000000004").fighter_1_id,
        fight(tables, "f000000000000004").fighter_2_id,
    }
    assert rep.resolved_by_rule == 1 and rep.resolved_by_override == 2
    # misspelled name resolved through the overrides file
    assert "c000000000000001" in set(tables["fights"].fighter_1_id) | set(
        tables["fights"].fighter_2_id
    )


def test_round_stats(built):
    tables, rep = built
    st = tables["fight_stats_round"]
    assert rep.fights_without_round_stats == 2  # blank stats rows (NC) and an old fight with none
    one = st[(st.fight_id == "f000000000000001") & (st.fighter_id == "b000000000000001")]
    assert list(one["round"]) == [1, 2]
    assert one.iloc[0].sig_str_landed == 10 and one.iloc[0].sig_str_att == 20
    assert one.iloc[0].control_s == 10 and one.iloc[0].knockdowns == 1
    cruz = st[st.fight_id == "f000000000000002"]
    assert cruz.control_s.isna().all()  # '--' is null, not 0
    assert len(st) == 2 * (2 + 3 + 1 + 1 + 3 + 1) and cruz["round"].max() == 3


def test_fighters_table(built):
    fr = built[0]["fighters"].set_index("fighter_id")
    assert fr.loc["b000000000000001", "height_cm"] == 165.1
    assert fr.loc["b000000000000001", "reach_cm"] == 162.6
    assert fr.loc["b000000000000001", "dob"] == "1990-01-02"
    assert pd.isna(fr.loc["a000000000000003", "height_cm"])
    assert fr.loc["c000000000000001", "name"] == "Nick Colley"
    assert "weight" not in fr.columns


def test_unresolved_name_fails(tmp_path):
    with pytest.raises(ValidationError, match="cannot be resolved"):
        build(FIX / "source", tmp_path, None)  # no overrides: Mike Davis / Nick Colly unresolved


def test_build_writes_and_validates(tmp_path):
    rep = build(FIX / "source", tmp_path, FIX / "overrides")
    assert rep.counts["fights"] == 8
    tables = load_tables(tmp_path)
    assert validate(tables)["fights_without_round_stats"] == 2


def test_validate_catches_problems(built):
    tables = {k: v.copy() for k, v in built[0].items()}
    tables["fights"].loc[0, "winner_id"] = "zzz"
    with pytest.raises(ValidationError, match="winner_id"):
        validate(tables)
    tables = {k: v.copy() for k, v in built[0].items()}
    tables["fight_stats_round"].loc[0, "sig_str_landed"] = 999
    with pytest.raises(ValidationError, match="landed"):
        validate(tables)
    tables = {k: v.copy() for k, v in built[0].items()}
    tables["fight_stats_round"] = tables["fight_stats_round"].iloc[1:]
    with pytest.raises(ValidationError, match="round counts"):
        validate(tables)
    tables = {k: v.copy() for k, v in built[0].items()}
    st = tables["fight_stats_round"]
    tables["fight_stats_round"] = st[~((st.fight_id == "f000000000000001") & (st["round"] == 2))]
    with pytest.raises(ValidationError, match="incomplete"):
        validate(tables)
    with pytest.raises(ValidationError, match="expected 7"):
        validate(built[0], expected_fights=7)
