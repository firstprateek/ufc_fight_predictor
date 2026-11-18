import math
from dataclasses import replace

import pandas as pd
import pytest

from ufc.features import (
    ELO_K,
    ELO_START,
    HALF_LIFE_FIGHTS,
    Imputation,
    PhysicalNullError,
    build_features,
    check_physical,
    check_truncation,
    feature_columns,
    fit_imputation,
    swap_orientation,
)

FIGHTERS = pd.DataFrame(
    [
        {"fighter_id": "A", "height_cm": 180.0, "reach_cm": 185.0, "stance": "Orthodox",
         "dob": "1990-01-01"},
        {"fighter_id": "B", "height_cm": 175.0, "reach_cm": None, "stance": "Southpaw",
         "dob": None},
        {"fighter_id": "C", "height_cm": None, "reach_cm": 170.0, "stance": None,
         "dob": "1995-06-15"},
    ]
)  # fmt: skip


def _fight(fid, day, f1, f2, winner, outcome, method, end_round, end_time_s, order=1):
    return {
        "fight_id": fid, "event_id": "e" + day, "date": day, "bout_order": order,
        "fighter_1_id": f1, "fighter_2_id": f2, "winner_id": winner, "outcome": outcome,
        "method": method, "end_round": end_round, "end_time_s": end_time_s,
        "weight_class": "Lightweight", "gender": "male",
    }  # fmt: skip


FIGHTS = pd.DataFrame(
    [
        _fight("f1", "2020-01-01", "A", "B", "A", "win", "KO/TKO", 1, 60),
        _fight("f2", "2020-02-01", "A", "C", "C", "win", "Submission", 2, 100),
        _fight("f3", "2020-02-01", "B", "C", None, "draw", "Decision - Majority", 3, 300, 2),
        _fight("f4", "2020-03-01", "A", "B", None, "nc", "Overturned", 3, 300),
        _fight("f5", "2020-04-01", "A", "C", "A", "win", "Decision - Unanimous", 3, 300),
        _fight("f6", "2020-05-01", "A", "C", "A", "win", "Decision - Unanimous", 3, 300),
    ]
).astype({"end_round": "Int64", "end_time_s": "Int64"})


def _stat(fid, who, rnd, sig_l, sig_a, td_l=0, td_a=0, sub=0, kd=0, ctrl=0):
    return {
        "fight_id": fid, "fighter_id": who, "round": rnd, "sig_str_landed": sig_l,
        "sig_str_att": sig_a, "td_landed": td_l, "td_att": td_a, "sub_attempts": sub,
        "knockdowns": kd, "control_s": ctrl,
    }  # fmt: skip


# f3 and f4 have no round stats (they feed experience features only)
STATS = pd.DataFrame(
    [
        _stat("f1", "A", 1, 30, 40, kd=1, ctrl=None),
        _stat("f1", "B", 1, 5, 10),
        _stat("f2", "A", 1, 10, 20, td_l=1, td_a=2, ctrl=30),
        _stat("f2", "A", 2, 10, 20, td_a=1, ctrl=None),
        _stat("f2", "C", 1, 20, 25, sub=2),
        _stat("f2", "C", 2, 20, 25, sub=1, ctrl=60),
        _stat("f5", "A", 1, 50, 100),
        _stat("f5", "C", 1, 10, 20),
    ]
)


@pytest.fixture(scope="module")
def feats():
    return build_features(FIGHTS, STATS, FIGHTERS).set_index("fight_id")


def test_one_row_per_fight_with_expected_columns(feats):
    assert list(feats.index) == ["f1", "f2", "f3", "f4", "f5", "f6"]
    for c in ("f1_elo", "f2_elo", "diff_elo", "f1_slpm_ewm", "diff_age", "f2_stance"):
        assert c in feats.columns
    assert "diff_stance" not in feats.columns


def test_first_fight_has_nulls_never_zero(feats):
    r = feats.loc["f1"]
    assert r.f1_n_fights == 0 and r.f2_n_fights == 0
    for c in ("win_rate", "n_wins", "win_streak", "days_since_last", "slpm_career", "str_acc_ewm"):
        assert pd.isna(r[f"f1_{c}"]) and pd.isna(r[f"diff_{c}"]), c
    assert r.f1_elo == r.f2_elo == ELO_START and r.diff_elo == 0


def test_only_strictly_earlier_fights_count(feats):
    # f2 and f3 share a date: neither sees the other. C has no history at f2 and f3.
    assert feats.loc["f2"].f2_n_fights == 0 and feats.loc["f3"].f2_n_fights == 0
    assert feats.loc["f3"].f1_n_fights == 1  # B saw f1 only
    assert feats.loc["f2"].f1_n_fights == 1  # A saw f1 only, not f3's date-mate f2
    assert feats.loc["f2"].f1_days_since_last == 31


def test_experience_features(feats):
    r = feats.loc["f5"]  # A: W (KO), L (sub), NC ; C: W (sub), D
    assert (r.f1_n_fights, r.f1_n_wins, r.f1_n_losses) == (3, 1, 1)
    assert r.f1_win_rate == pytest.approx(1 / 3)
    assert r.f1_win_streak == 0 and r.f1_loss_streak == 0  # the NC broke both
    assert r.f1_ko_win_share == 1.0 and r.f1_sub_win_share == 0.0 and r.f1_ko_loss_share == 0.0
    assert r.f1_days_since_last == 31
    assert (r.f2_n_fights, r.f2_n_wins, r.f2_win_streak) == (2, 1, 0)  # C: W then draw
    assert r.f2_sub_win_share == 1.0
    assert r.diff_n_fights == 1
    r = feats.loc["f4"]  # A after W, L: loss streak 1; B after L, D
    assert r.f1_loss_streak == 1 and r.f1_win_streak == 0
    assert r.f1_ko_loss_share == 0.0 and r.f2_ko_loss_share == 1.0


def test_physical_features(feats):
    r = feats.loc["f1"]
    assert r.f1_age == pytest.approx(
        (pd.Timestamp("2020-01-01") - pd.Timestamp("1990-01-01")).days / 365.25
    )
    imp = fit_imputation(FIGHTS, FIGHTERS)
    assert r.f2_age == imp.age and r.diff_age == pytest.approx(r.f1_age - imp.age)
    assert r.diff_height_cm == 5.0
    assert r.f2_reach_cm == pytest.approx(175.0 * imp.reach_ratio)  # B: imputed from height
    assert (r.f1_stance, r.f2_stance, r.f1_southpaw, r.f2_southpaw, r.diff_southpaw) == (
        "Orthodox", "Southpaw", 0.0, 1.0, -1.0,
    )  # fmt: skip
    c = feats.loc["f2"]  # C lacks height and stance: group median, Orthodox
    assert c.f2_height_cm == 177.5 and c.f2_reach_cm == 170.0
    assert (c.f2_stance, c.f2_southpaw) == ("Orthodox", 0.0)


def test_imputation_constants():
    imp = fit_imputation(FIGHTS, FIGHTERS)
    assert imp.height_global == 177.5 and imp.stance == "Orthodox"
    assert imp.height_group == {("Lightweight", "male"): 177.5}
    assert imp.reach_ratio == pytest.approx(185.0 / 180.0)


def test_height_imputation_falls_back_to_gender_then_global():
    imp = Imputation(
        height_global=170.0,
        height_gender={"female": 165.0},
        height_group={("Strawweight", "female"): 160.0},
        reach_ratio=1.0,
        age=30.0,
        stance="Orthodox",
        group_of={
            "s": ("Strawweight", "female"),
            "f": ("Flyweight", "female"),
            "m": ("Welterweight", "male"),
        },
    )
    assert (imp.height("s"), imp.height("f"), imp.height("m")) == (160.0, 165.0, 170.0)
    assert imp.height("never fought") == 170.0


def test_group_is_the_most_frequent_division():
    fights = FIGHTS.copy()
    fights.loc[fights.fight_id == "f1", "weight_class"] = "Welterweight"
    assert fit_imputation(fights, FIGHTERS).group_of["B"] == ("Lightweight", "male")


def test_missing_snapshot_fields_leave_no_trace_in_the_row():
    # C lacks height, reach, stance and dob. Its feature row must equal the row built when those
    # are replaced by the imputed values: nothing else may betray that they were missing.
    bare = FIGHTERS.copy()
    bare.loc[bare.fighter_id == "C", ["height_cm", "reach_cm", "stance", "dob"]] = None
    fights = FIGHTS[FIGHTS.fight_id == "f2"]  # A vs C, one date
    stats = STATS[STATS.fight_id == "f2"]
    imp = replace(fit_imputation(FIGHTS, bare), age=10000 / 365.25)  # exactly 10000 days old
    a = build_features(fights, stats, bare, imp)
    check_physical(a)
    h = imp.height("C")
    filled = bare.copy()
    filled.loc[filled.fighter_id == "C", ["height_cm", "reach_cm", "stance", "dob"]] = [
        h, h * imp.reach_ratio, imp.stance, str(pd.Timestamp("2020-02-01") - pd.Timedelta(days=10000))[:10],
    ]  # fmt: skip
    b = build_features(fights, stats, filled, imp)
    pd.testing.assert_frame_equal(a, b)
    assert a.f2_height_cm[0] == h and a.f2_stance[0] == "Orthodox"
    # whole fixture: no physical null anywhere, and the rest of C's rows is the same
    full = build_features(FIGHTS, STATS, bare, imp)
    check_physical(full)


def test_physical_null_guard():
    df = build_features(FIGHTS, STATS, FIGHTERS)
    check_physical(df)  # clean
    for col in ("f1_reach_cm", "f2_stance", "diff_age", "f1_southpaw"):
        bad = df.copy()
        bad[col] = bad[col].astype(object)
        bad.loc[0, col] = None
        with pytest.raises(PhysicalNullError, match=col):
            check_physical(bad)


def test_cli_features_fails_on_a_physical_null(tmp_path, monkeypatch, capsys):
    from pathlib import Path

    import ufc.cli as cli
    from ufc.build import build

    fix = Path(__file__).parent / "fixtures"
    build(fix / "source", tmp_path / "processed", fix / "overrides")
    real = cli.build_features

    def nulled(*a, **k):
        df = real(*a, **k)
        df["f1_reach_cm"] = None
        return df

    monkeypatch.setattr(cli, "build_features", nulled)
    assert cli.main(["--data-dir", str(tmp_path), "features"]) == 1
    assert "physical" in capsys.readouterr().err
    assert not (tmp_path / "processed" / "features.parquet").exists()


def test_career_rates(feats):
    r = feats.loc["f2"]  # A's only prior fight: f1, 60 s, 30/40 sig, 1 kd; B's opp landed 5/10
    assert r.f1_slpm_career == pytest.approx(30.0)
    assert r.f1_sapm_career == pytest.approx(5.0)
    assert r.f1_str_acc_career == pytest.approx(0.75)
    assert r.f1_str_def_career == pytest.approx(0.5)
    assert r.f1_kd15_career == pytest.approx(15.0)  # 1 knockdown in 1 minute
    assert pd.isna(r.f1_td_acc_career)  # no takedown attempts: null, not 0
    assert pd.isna(r.f1_ctrl_share_career)  # control seconds were null in that fight
    r = feats.loc["f5"]  # A: f1 (60 s) and f2 (400 s); f3/f4 have no stats
    assert r.f1_slpm_career == pytest.approx((30 + 20) / ((60 + 400) / 60))
    assert r.f1_td_acc_career == pytest.approx(1 / 3)
    assert pd.isna(r.f1_td_def_career)  # C attempted no takedowns
    assert r.f1_ctrl_share_career == pytest.approx(30 / 400)  # f1 had no control data: skipped
    assert r.f1_sub15_career == pytest.approx(0.0)  # a real 0, not a null


def test_exponentially_weighted_rates(feats):
    # A's prior fights before f6, oldest first: f1, f2, f4 (no stats, still takes a slot), f5.
    # So f1 is 3 fights back, f2 is 2 back, f5 is the most recent (weight 1).
    r = feats.loc["f6"]
    w1, w2, w5 = 0.5 ** (3 / HALF_LIFE_FIGHTS), 0.5 ** (2 / HALF_LIFE_FIGHTS), 1.0
    assert r.f1_str_acc_ewm == pytest.approx(
        (w1 * 30 + w2 * 20 + w5 * 50) / (w1 * 40 + w2 * 40 + w5 * 100)
    )
    one = feats.loc["f2"]  # a single prior fight: weighted and career agree
    assert one.f1_slpm_ewm == pytest.approx(one.f1_slpm_career)


def _expected(r1, r2):
    return 1 / (1 + 10 ** ((r2 - r1) / 400))


def test_elo_updates(feats):
    a1 = ELO_START + ELO_K * (1 - 0.5)  # f1: A beats B between equals
    b1 = ELO_START - ELO_K * 0.5
    assert feats.loc["f2"].f1_elo == pytest.approx(a1) and feats.loc["f2"].f2_elo == ELO_START
    # f3 is on f2's date: it uses ratings from before that date, so C is still 1500
    assert feats.loc["f3"].f1_elo == pytest.approx(b1) and feats.loc["f3"].f2_elo == ELO_START
    e_c = _expected(ELO_START, a1)  # f2: C beats A
    c2 = ELO_START + ELO_K * (1 - e_c)
    a2 = a1 - ELO_K * (1 - e_c)
    e_draw = _expected(b1, ELO_START)  # f3: draw is half a win
    b3 = b1 + ELO_K * (0.5 - e_draw)
    c3 = c2 + ELO_K * (0.5 - (1 - e_draw))
    r = feats.loc["f4"]
    assert r.f1_elo == pytest.approx(a2) and r.f2_elo == pytest.approx(b3)
    # f4 is a no contest: ratings unchanged going into f5; C carries both f2 and f3 updates
    assert feats.loc["f5"].f1_elo == pytest.approx(a2)
    assert feats.loc["f5"].f2_elo == pytest.approx(c3)
    assert r.diff_elo == pytest.approx(a2 - b3)


def test_diffs_are_f1_minus_f2(feats):
    for c in feature_columns(feats):
        if c.startswith("f1_") and c != "f1_stance":
            d = feats[c] - feats["f2_" + c[3:]]
            pd.testing.assert_series_equal(d, feats["diff_" + c[3:]], check_names=False,
                                           check_dtype=False)  # fmt: skip


def test_truncation_invariance():
    full = build_features(FIGHTS, STATS, FIGHTERS)
    # every fight dated on or after f5, f4 or f2 removed (the fight itself kept)
    assert check_truncation(FIGHTS, STATS, FIGHTERS, full, sample=6) == []


def test_truncation_check_catches_a_leaky_builder(monkeypatch):
    import ufc.features.build as B

    real = B.build_features

    def leaky(fights, stats, fighters, imputation=None):  # depends on how many fights exist
        df = real(fights, stats, fighters, imputation)
        df["f1_elo"] = df["f1_elo"] + len(fights)
        return df

    monkeypatch.setattr(B, "build_features", leaky)
    full = B.build_features(FIGHTS, STATS, FIGHTERS)
    assert len(check_truncation(FIGHTS, STATS, FIGHTERS, full, sample=6)) >= 4


def test_features_unchanged_by_later_results():
    # Flip the winner of the last fight: nothing before it may change.
    flipped = FIGHTS.copy()
    flipped.loc[flipped.fight_id == "f5", "winner_id"] = "C"
    a = build_features(FIGHTS, STATS, FIGHTERS).set_index("fight_id")
    b = build_features(flipped, STATS, FIGHTERS).set_index("fight_id")
    pd.testing.assert_frame_equal(a.loc[["f1", "f2", "f3", "f4"]], b.loc[["f1", "f2", "f3", "f4"]])


def test_features_do_not_use_the_fights_own_result_or_stats():
    # The truncation check keeps the fight itself, so this property is tested separately.
    flipped = FIGHTS.copy()
    flipped.loc[flipped.fight_id == "f5", ["winner_id", "method"]] = ["C", "KO/TKO"]
    stats = STATS[STATS.fight_id != "f5"]
    a = build_features(FIGHTS, STATS, FIGHTERS).set_index("fight_id").loc["f5"]
    b = build_features(flipped, stats, FIGHTERS).set_index("fight_id").loc["f5"]
    pd.testing.assert_series_equal(a, b)


def test_orientation_swap(feats):
    df = feats.reset_index()
    sw = swap_orientation(df)
    assert (sw.fighter_1_id == df.fighter_2_id).all()
    assert (sw.f1_elo == df.f2_elo).all() and (sw.f2_elo == df.f1_elo).all()
    pd.testing.assert_series_equal(sw.diff_elo, -df.diff_elo)
    pd.testing.assert_frame_equal(swap_orientation(sw), df)
    assert sw.loc[0, "diff_age"] == -df.loc[0, "diff_age"] and not math.isnan(sw.loc[0, "diff_age"])


def test_cli_features_check(tmp_path, capsys):
    from pathlib import Path

    from ufc.build import build
    from ufc.cli import main

    fix = Path(__file__).parent / "fixtures"
    build(fix / "source", tmp_path / "processed", fix / "overrides")
    assert main(["--data-dir", str(tmp_path), "features", "--check", "--sample", "10"]) == 0
    out = capsys.readouterr().out
    assert "features: 8 rows" in out and "truncation invariance holds" in out
    df = pd.read_parquet(tmp_path / "processed" / "features.parquet")
    assert len(df) == 8 and df.fight_id.is_unique
