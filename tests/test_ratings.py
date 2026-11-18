import dataclasses
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from test_backtest import synthetic

import ufc.features.tune as TUNE
from ufc.backtest.run import scored_fights, walk_forward
from ufc.features import (
    DIMENSIONS,
    RATING_PARAMS,
    build_features,
    check_physical,
    check_truncation,
    column_group,
)
from ufc.features.adjusted import Adjusted
from ufc.features.bout import Bout, Card, method_category
from ufc.features.context import Context
from ufc.features.params import RatingParams
from ufc.features.ratings import Glicko, MethodElo
from ufc.features.tune import TuneCutoffError, make_days, tune_ratings

P = RatingParams(
    glicko_c=0.0, welo_k=32, welo_unan=1.0, welo_ko=2.0, welo_sub=1.0, welo_other=0.5,
    adj_k0=1.0, adj_rate_min=0.0,
)  # fmt: skip
NAN = float("nan")


def card(f1, f2, **kw):
    d = dict(weight_class="Lightweight", scheduled_rounds=3.0, is_title=False, elo1=NAN, elo2=NAN)
    return Card("x", f1, f2, **{**d, **kw})


def bout(f1, f2, s1, day=date(2020, 1, 1), method="ko", m1=None, m2=None, **kw):
    return Bout(card(f1, f2, **kw), day, s1, method, m1 or {}, m2 or {})


def test_glicko_matches_glickmans_worked_example():
    g = Glicko(c=0.0)
    for fid, (r, rd) in {
        "P": (1500, 200),
        "a": (1400, 30),
        "b": (1550, 100),
        "c": (1700, 300),
    }.items():
        g.r[fid], g.rd[fid], g.last[fid] = r, rd, date(2019, 1, 1)
    on = date(2020, 1, 1)
    g.update(on, [bout("P", "a", 1.0), bout("P", "b", 0.0), bout("P", "c", 0.0)])
    assert g.r["P"] == pytest.approx(1464.1, abs=0.1)
    assert g.rd["P"] == pytest.approx(151.4, abs=0.1)


def test_glicko_deviation_grows_with_days_and_is_capped():
    g = Glicko(c=5.0)
    assert g.state("new", date(2020, 1, 1)) == (1500.0, 350.0)  # debutant: the prior
    g.update(date(2020, 1, 1), [bout("A", "B", 1.0)])
    _, rd0 = g.rd["A"], g.rd["A"]
    r, rd = g.state("A", date(2020, 1, 1))
    assert rd == pytest.approx(rd0)
    _, rd100 = g.state("A", date(2020, 4, 10))  # 100 days later
    assert rd100 == pytest.approx(math.sqrt(rd0**2 + 25 * 100))
    assert g.state("A", date(2040, 1, 1))[1] == 350.0
    # a no contest changes nothing and does not reset the clock
    g.update(date(2020, 6, 1), [bout("A", "C", None)])
    assert g.last["A"] == date(2020, 1, 1) and "C" not in g.r


def test_glicko_expected_is_symmetric_and_uses_both_deviations():
    g = Glicko(c=0.0)
    g.r.update(A=1600.0, B=1500.0)
    g.rd.update(A=50.0, B=50.0)
    g.last.update(A=date(2019, 1, 1), B=date(2019, 1, 1))
    on = date(2020, 1, 1)
    e = g.expected("A", "B", on)
    assert e > 0.5 and e + g.expected("B", "A", on) == pytest.approx(1.0)
    g.rd["B"] = 300.0  # a more uncertain opponent pulls the expectation towards 0.5
    assert 0.5 < g.expected("A", "B", on) < e


def test_method_elo_scales_the_update_by_method():
    m = MethodElo(P)
    m.update(date(2020, 1, 1), [bout("A", "B", 1.0, method="ko")])
    assert m.rating("A") == pytest.approx(1500 + 32 * 2.0 * 0.5)
    assert m.rating("B") == pytest.approx(1500 - 32 * 2.0 * 0.5)
    m = MethodElo(P)
    m.update(
        date(2020, 1, 1), [bout("A", "B", 1.0, method="split"), bout("C", "D", 0.5, method="other")]
    )
    assert m.rating("A") == pytest.approx(1516) and m.rating("C") == 1500.0
    m = MethodElo(P)
    m.update(date(2020, 1, 1), [bout("A", "B", 0.0, method="other"), bout("C", "D", None)])
    assert m.rating("A") == pytest.approx(1500 - 32 * 0.5 * 0.5) and "C" not in m.r


def test_method_categories():
    assert method_category("Decision - Majority") == method_category("Decision - Split") == "split"
    assert method_category("Decision - Unanimous") == "unanimous"
    assert method_category("TKO - Doctor's Stoppage") == "ko" == method_category("KO/TKO")
    assert method_category("Submission") == "sub"
    assert {method_category(m) for m in ("DQ", "Overturned", "Could Not Continue", "Other")} == {
        "other"
    }


def _scaled(a: Adjusted, n=100, mean=5.0, sd=2.0):
    for d in DIMENSIONS:
        a.run[d].n, a.run[d].mean, a.run[d].m2 = n, mean, n * sd * sd


def test_adjusted_update_signs_rates_and_expectation():
    a = Adjusted(P)  # rate = 1 / (fights_with_stats + 1)
    _scaled(a)
    f = Card("x", "A", "B", "Lightweight", 3.0, False, NAN, NAN)
    # A lands 9 (z = +2), B lands 5 (z = 0)
    a.update(date(2020, 1, 1), [bout("A", "B", 1.0, m1={"slpm": 9.0}, m2={"slpm": 5.0})])
    fa, fb = a.features("A", date(2020, 2, 1), f), a.features("B", date(2020, 2, 1), f)
    assert fa["adj_slpm_off"] == pytest.approx(2.0)  # rate 1 x error 2
    assert fb["adj_slpm_def"] == pytest.approx(-2.0)  # B conceded more than expected
    assert fa["adj_slpm_def"] == fb["adj_slpm_off"] == 0.0  # B's zero error moves nothing else
    assert fa["adj_td15_off"] == 0.0  # other dimensions untouched
    e1, e2 = a.pair(f, date(2020, 2, 1))["adj_slpm_exp"]
    mean, sd = a.run["slpm"].scale()  # the league mean and sd now include that fight
    assert e1 == pytest.approx(mean + sd * (2.0 - (-2.0)))  # off_A - def_B
    assert e2 == pytest.approx(mean + sd * (0.0 - 0.0))  # off_B - def_A
    # the next update uses rate 1 / (1 + 1) for both fighters
    mean2, sd2 = a.run["slpm"].scale()
    err_a = (5.0 - mean2) / sd2 - (2.0 - (-2.0))
    a.update(date(2020, 3, 1), [bout("A", "B", 1.0, m1={"slpm": 5.0}, m2={"slpm": 5.0})])
    assert a.off["slpm"]["A"] == pytest.approx(2.0 + 0.5 * err_a)
    assert a.defn["slpm"]["B"] == pytest.approx(-2.0 - 0.5 * err_a)


def test_adjusted_rate_floor_and_no_update_without_stats_or_before_min_obs():
    a = Adjusted(RatingParams(**{**P.__dict__, "adj_rate_min": 0.3, "adj_k0": 50.0}))
    _scaled(a)
    a.update(date(2020, 1, 1), [bout("A", "B", 1.0, m1={"slpm": 9.0}, m2={"slpm": 5.0})])
    assert a.off["slpm"]["A"] == pytest.approx(0.3 * 2.0)  # floor beats 1/(0+50)
    before = a.off["slpm"]["A"]
    a.update(date(2020, 2, 1), [bout("A", "C", 1.0)])  # no round stats
    assert a.off["slpm"]["A"] == before and a.n["A"] == 1
    fresh = Adjusted(P)  # fewer than MIN_OBS observations: scale undefined, nothing moves
    fresh.update(date(2020, 1, 1), [bout("A", "B", 1.0, m1={"slpm": 9.0}, m2={"slpm": 5.0})])
    assert fresh.off["slpm"]["A"] == 0.0 and fresh.run["slpm"].n == 2


def test_adjusted_same_date_fights_use_pre_date_state():
    def run(order):
        a = Adjusted(P)
        _scaled(a)
        bs = [
            bout("A", "B", 1.0, m1={"slpm": 9.0}, m2={"slpm": 5.0}),
            bout("A", "C", 1.0, m1={"slpm": 1.0}, m2={"slpm": 7.0}),
        ]
        a.update(date(2020, 1, 1), [bs[i] for i in order])
        return {d: dict(a.off[d]) for d in DIMENSIONS}, {d: dict(a.defn[d]) for d in DIMENSIONS}

    assert run([0, 1]) == run([1, 0])


def test_trackers_are_independent_of_the_order_of_fights_within_a_date():
    """One fighter in two bouts on a date (old tournament nights): the update is order independent."""
    on = date(2020, 1, 1)
    bs = [
        bout("A", "B", 1.0, method="ko", m1={"slpm": 9.0}, m2={"slpm": 5.0}),
        bout("A", "C", 0.0, method="sub", m1={"slpm": 2.0}, m2={"slpm": 7.0}),
        bout("D", "A", 0.5, method="unanimous", m1={"slpm": 4.0}, m2={"slpm": 4.0}),
    ]

    def states(order):
        g, m, a, c = Glicko(6.0), MethodElo(P), Adjusted(P), Context()
        _scaled(a)
        elo = dict.fromkeys("ABCD", 1500.0)
        for t in (g, m, a, c):
            t.update(on, [bs[i] for i in order], elo)
        return g.r, g.rd, m.r, a.off, a.defn, a.n

    ref = states([0, 1, 2])
    for order in ([2, 1, 0], [1, 0, 2]):
        got = states(order)
        assert got[0] == pytest.approx(ref[0]) and got[1] == pytest.approx(ref[1])
        assert got[2] == pytest.approx(ref[2]) and dict(got[5]) == dict(ref[5])
        for d in DIMENSIONS:
            assert dict(got[3][d]) == pytest.approx(dict(ref[3][d]))
            assert dict(got[4][d]) == pytest.approx(dict(ref[4][d]))


def _stats(fights, seed=2):
    """Two rows (one round) per fight with random striking, takedown, control and submission stats."""
    rng = np.random.default_rng(seed)
    rows = []
    for r in fights.itertuples():
        for who in (r.fighter_1_id, r.fighter_2_id):
            att = int(rng.integers(5, 60))
            td_a = int(rng.integers(0, 6))
            rows.append(
                {
                    "fight_id": r.fight_id, "fighter_id": who, "round": 1,
                    "sig_str_landed": int(rng.integers(0, att + 1)), "sig_str_att": att,
                    "td_landed": int(rng.integers(0, td_a + 1)), "td_att": td_a,
                    "sub_attempts": int(rng.integers(0, 3)), "knockdowns": int(rng.integers(0, 2)),
                    "control_s": float(rng.integers(0, 60)),
                }
            )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def world():
    fights, fighters = synthetic(1000)
    stats = _stats(fights)
    return fights, stats, fighters, build_features(fights, stats, fighters)


def test_new_columns_exist_and_follow_the_null_rules(world):
    fights, stats, fighters, feats = world
    groups = {column_group(c) for c in feats.columns}
    assert {"ratings", "adjusted", "context"} <= groups
    check_physical(feats)
    always = [c for c in feats.columns if column_group(c) in ("ratings", "adjusted")]
    assert not feats[always].isna().any().any()  # debutant priors are definitions, not nulls
    first = feats.sort_values("date").iloc[0]
    assert first.f1_glicko_r == 1500.0 and first.f1_glicko_rd == 350.0 and first.f1_welo == 1500.0
    assert first.f1_adj_slpm_off == 0.0 and first.diff_glicko_exp == 0.0
    assert pd.isna(first.f1_ctx_sos_mean) and first.f1_ctx_wc_first == 1.0
    assert first.f1_ctx_fights_24m == 0.0 and first.f1_ctx_wc_fights == 0.0
    # a fighter with prior fights has the history-based context
    later = feats[feats.f1_n_fights >= 3]
    assert not later.filter(like="ctx_sos_mean").isna().any().any()
    # diffs are f1 minus f2, also for the pair columns
    np.testing.assert_allclose(
        feats.diff_adj_slpm_exp, feats.f1_adj_slpm_exp - feats.f2_adj_slpm_exp, atol=1e-9
    )
    np.testing.assert_allclose(feats.f1_glicko_exp + feats.f2_glicko_exp, 1.0)


def test_truncation_invariance_holds_for_the_new_features(world):
    fights, stats, fighters, feats = world
    assert check_truncation(fights, stats, fighters, feats, sample=40, seed=1) == []


def test_new_features_ignore_a_fights_own_result_and_stats_and_later_results(world):
    fights, stats, fighters, feats = world
    fid = fights.fight_id.iloc[200]
    day = fights.set_index("fight_id").date[fid]
    flipped = fights.copy()
    i = flipped.fight_id == fid
    flipped.loc[i, ["winner_id", "method"]] = [
        flipped.loc[i, "fighter_2_id"].iloc[0]
        if flipped.loc[i, "winner_id"].iloc[0] == flipped.loc[i, "fighter_1_id"].iloc[0]
        else flipped.loc[i, "fighter_1_id"].iloc[0],
        "Submission",
    ]
    stats2 = stats[stats.fight_id != fid]  # also drop its stats
    other = build_features(flipped, stats2, fighters).set_index("fight_id")
    f = feats.set_index("fight_id")
    cols = [c for c in feats.columns if column_group(c) != "base"]
    pd.testing.assert_series_equal(f.loc[fid, cols], other.loc[fid, cols])
    # every fight dated on or before that day is unchanged by the changed result
    up_to = f.date <= day
    pd.testing.assert_frame_equal(f.loc[up_to, cols], other.loc[up_to, cols])
    assert not f.loc[~up_to, cols].equals(other.loc[~up_to, cols])  # later fights do see it


def test_same_date_fights_use_pre_date_state_even_for_a_repeated_fighter():
    fights, fighters = synthetic(30)
    stats = _stats(fights)
    day = fights.date.iloc[0]
    # fights x00000..x00002 share the first event; make fighter f1 of x00000 also fight in x00001
    two = fights.iloc[:2].copy()
    two["date"], two["event_id"] = day, "e1"
    two.loc[1, "fighter_1_id"] = two.loc[0, "fighter_1_id"]
    two.loc[1, "fighter_2_id"] = "f23" if two.loc[0, "fighter_2_id"] != "f23" else "f22"
    nxt = fights.iloc[2:3].copy()
    nxt["date"], nxt["event_id"], nxt["fighter_1_id"], nxt["fighter_2_id"] = (
        "2012-03-01", "e2", two.loc[0, "fighter_1_id"], "f21",
    )  # fmt: skip
    nxt["bout_order"] = 1
    both = pd.concat([two, nxt], ignore_index=True)
    both["winner_id"] = both["fighter_1_id"]
    both["bout_order"] = [1, 2, 1]
    swapped = both.copy()
    swapped["bout_order"] = [2, 1, 1]  # the same date's fights in the other order
    a = build_features(both, stats, fighters).set_index("fight_id")
    b = build_features(swapped, stats, fighters).set_index("fight_id")
    f = two.loc[0, "fighter_1_id"]
    for fid in two.fight_id:  # both fights of the date see the fighter's prior state
        r = a.loc[fid]
        side = "f1" if r.fighter_1_id == f else "f2"
        assert r[f"{side}_glicko_r"] == 1500.0 and r[f"{side}_welo"] == 1500.0
        assert r[f"{side}_adj_slpm_off"] == 0.0 and r[f"{side}_ctx_wc_fights"] == 0.0
    # after the date the fighter carries both fights' updates, whatever their order
    after = nxt.fight_id.iloc[0]
    cols = [
        c
        for c in a.columns
        if column_group(c) in ("ratings", "adjusted") or c in ("f1_elo", "f2_elo")
    ]
    pd.testing.assert_series_equal(a.loc[after, cols], b.loc[after, cols])
    assert a.loc[after, "f1_glicko_r"] != 1500.0


def test_context_features_by_hand():
    def fight(fid, day, f1, f2, w, wc, method="Decision - Unanimous", rounds=3, title=False):
        return {
            "fight_id": fid, "event_id": "e" + day, "date": day, "bout_order": 1,
            "fighter_1_id": f1, "fighter_2_id": f2, "winner_id": w, "outcome": "win",
            "method": method, "end_round": 3, "end_time_s": 300, "weight_class": wc,
            "gender": "male", "scheduled_rounds": rounds, "is_title": title,
        }  # fmt: skip

    fights = pd.DataFrame(
        [
            fight("d1", "2020-01-01", "A", "B", "A", "Lightweight", rounds=5, title=True),
            fight("d2", "2020-06-01", "A", "C", "C", "Welterweight", method="Submission"),
            fight("d3", "2022-03-01", "A", "B", "A", "Catch Weight"),
            fight("d4", "2023-01-01", "A", "C", "A", "Lightweight", rounds=5),
        ]
    ).astype({"end_round": "Int64", "end_time_s": "Int64", "scheduled_rounds": "Int64"})
    fighters = pd.DataFrame(
        [
            {"fighter_id": x, "height_cm": 180.0, "reach_cm": 180.0, "stance": "Orthodox",
             "dob": "1990-01-01"}
            for x in "ABC"
        ]
    )  # fmt: skip
    nostats = pd.DataFrame(columns=["fight_id", "fighter_id", "round"])
    f = build_features(fights, nostats, fighters).set_index("fight_id")
    d2, d3, d4 = f.loc["d2"], f.loc["d3"], f.loc["d4"]
    # d2: A's one prior fight was a unanimous decision win over B (both 1500), five rounds, a title
    assert d2.f1_ctx_sos_mean == 1500 and d2.f1_ctx_best_win_elo == 1500
    assert d2.f1_ctx_last_result == 1.0 and d2.f1_ctx_last_finish == 0.0
    assert d2.f1_ctx_five_round_fights == 1.0 and d2.f1_ctx_title_fights == 1.0
    assert d2.f1_ctx_wc_first == 1.0 and d2.f1_ctx_wc_move == 1.0  # Lightweight -> Welterweight
    assert d2.f1_ctx_elo_change3 == pytest.approx(d2.f1_elo - 1500)
    assert d2.f1_ctx_is_five_rounds == 0.0 and d2.f1_ctx_is_title == 0.0
    assert pd.isna(d2.f2_ctx_sos_mean) and d2.f2_ctx_fights_24m == 0
    # d3 (catch weight): A lost by submission last time, to C, whose Elo was 1500 before d2
    assert d3.f1_ctx_last_result == -1.0 and d3.f1_ctx_last_finish == 1.0
    assert d3.f1_ctx_last_opp_elo == 1500 and d3.f1_ctx_sos_win_mean == 1500
    assert pd.isna(d3.f1_ctx_wc_move)  # catch weight has no place in the order
    assert d3.f1_ctx_fights_24m == 1.0  # d1 (2020-01) is older than 24 months, d2 is not
    peak = max(1500, d2.f1_elo)  # A's Elo after d1 is f1_elo on d2
    assert d3.f1_ctx_elo_peak_gap == pytest.approx(peak - d3.f1_elo)
    # d4: back to Lightweight after Welterweight (last ranked division), one prior Lightweight fight
    assert d4.f1_ctx_wc_move == -1.0 and d4.f1_ctx_wc_fights == 1.0 and d4.f1_ctx_wc_first == 0.0
    assert d4.f1_ctx_is_five_rounds == 1.0 and d4.f1_ctx_five_round_fights == 1.0
    assert d4.f1_ctx_fights_24m == 1.0  # d3 only: d2 is more than 24 months back
    assert d4.f1_ctx_elo_change3 == pytest.approx(d4.f1_elo - 1500)  # three prior fights


def test_context_uses_prefight_opponent_elo_not_current():
    c = Context()
    cd = Card("x", "A", "B", "Lightweight", 3.0, False, 1510.0, 1490.0)
    c.update(date(2020, 1, 1), [Bout(cd, date(2020, 1, 1), 1.0, "ko")], {"A": 1530.0, "B": 1470.0})
    out = c.features("B", date(2021, 1, 1), card("B", "C", elo1=1400.0, elo2=1500.0))
    assert out["ctx_last_opp_elo"] == 1510.0  # A's rating before that fight, not 1530


# --- tuning ---------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def tuning_world():
    fights, fighters = synthetic(1000)  # 2012 to mid 2017
    return fights, _stats(fights)


def test_tuning_refuses_a_cutoff_inside_the_test_period(tuning_world):
    fights, stats = tuning_world
    with pytest.raises(TuneCutoffError):
        tune_ratings(fights, stats, before="2015-01-02")


def test_tuning_ignores_everything_dated_on_or_after_the_cutoff(tuning_world, monkeypatch):
    fights, stats = tuning_world
    seen = []
    real = TUNE.make_days

    def spy(f, s):
        seen.append((f.date.max(), s.fight_id.isin(f.fight_id).all()))
        return real(f, s)

    monkeypatch.setattr(TUNE, "make_days", spy)
    p, diag = tune_ratings(fights, stats, before="2015-01-01")
    assert seen == [(seen[0][0], True)] and seen[0][0] < "2015-01-01"
    assert diag["fights"] == (fights.date < "2015-01-01").sum() > 300
    # corrupt everything from the cutoff on: results, stats, even the dates' methods
    late = fights.date >= "2015-01-01"
    bad = fights.copy()
    bad.loc[late, "winner_id"] = bad.loc[late, "fighter_1_id"]
    bad.loc[late, "method"] = "Submission"
    bad_stats = stats.copy()
    bad_stats.loc[bad_stats.fight_id.isin(fights.fight_id[late]), "sig_str_landed"] = 0
    p2, diag2 = tune_ratings(bad, bad_stats, before="2015-01-01")
    assert p2 == p and diag2 == diag
    # an earlier cutoff is allowed and uses fewer fights
    assert tune_ratings(fights, stats, before="2014-01-01")[1]["fights"] < diag["fights"]


def test_tuning_is_deterministic_and_scores_one_step_ahead(tuning_world):
    fights, stats = tuning_world
    a = tune_ratings(fights, stats)
    assert a == tune_ratings(fights, stats)
    # a tracker that has seen a fight cannot have predicted it from the result: the first
    # prediction of any fighter pair is 0.5 for a fresh tracker
    days = make_days(fights[fights.date < "2015-01-01"], stats)
    g = TUNE._GlickoPred(6.0)
    on, bouts = days[0]
    assert g.predict(bouts[0].card, on) == 0.5


def test_tuner_predicts_each_fight_before_learning_it(tuning_world):
    fights, stats = tuning_world
    sub, sub_stats = TUNE.pre_cutoff(fights, stats, "2013-01-01")
    days = make_days(sub, sub_stats)

    class Recorder(TUNE._GlickoPred):
        def __init__(self):
            super().__init__(6.0)
            self.log, self.preds = [], []

        def predict(self, card, on):
            p = super().predict(card, on)
            self.log.append(("predict", on))
            self.preds.append((card.fight_id, p))
            return p

        def update(self, on, bouts, elo_after=None):
            self.log.append(("update", on))
            super().update(on, bouts, elo_after)

    rec = Recorder()
    TUNE._log_loss(rec, days, date(2000, 1, 1))
    # per date: every prediction comes before that date's update
    assert [e for e, _ in rec.log].count("update") == len(days)
    seen_update = set()
    for kind, on in rec.log:
        if kind == "update":
            seen_update.add(on)
        else:
            assert on not in seen_update
    # the result of a fight (and everything after) changes no prediction made up to and including it
    k = len(days) // 2
    flipped = list(days)
    on, bouts = flipped[k]
    flipped[k] = (
        on,
        [dataclasses.replace(b, s1=None if b.s1 is None else 1.0 - b.s1) for b in bouts],
    )
    rec2 = Recorder()
    TUNE._log_loss(rec2, flipped, date(2000, 1, 1))
    n_through = sum(len(b) for _, b in days[: k + 1])
    assert rec.preds[:n_through] == rec2.preds[:n_through]
    assert rec.preds[n_through:] != rec2.preds[n_through:]


def test_committed_constants_are_the_default_in_the_feature_build():
    from dataclasses import replace

    fights, fighters = synthetic(120)
    stats = _stats(fights)
    default = build_features(fights, stats, fighters)
    pd.testing.assert_frame_equal(
        default, build_features(fights, stats, fighters, params=RATING_PARAMS)
    )
    other = build_features(fights, stats, fighters, params=replace(RATING_PARAMS, glicko_c=0.0))
    assert not default.f1_glicko_rd.equals(other.f1_glicko_rd)
    assert default.f1_elo.equals(other.f1_elo)  # plain Elo does not use the tuned constants


def test_cli_tune_ratings_compares_with_the_committed_constants(tmp_path, capsys, monkeypatch):
    from dataclasses import replace
    from pathlib import Path

    import ufc.cli as CLI
    from ufc.build import build

    fix = Path(__file__).parent / "fixtures"
    build(fix / "source", tmp_path / "processed", fix / "overrides")
    args = ["--data-dir", str(tmp_path), "tune-ratings"]
    monkeypatch.setattr(CLI, "tune_ratings", lambda f, s, b: (RATING_PARAMS, _DIAG))
    assert CLI.main(args) == 0 and "matches the committed constants" in capsys.readouterr().out
    monkeypatch.setattr(
        CLI, "tune_ratings", lambda f, s, b: (replace(RATING_PARAMS, welo_k=1), _DIAG)
    )
    assert CLI.main(args) == 1 and "DIFFERS" in capsys.readouterr().err
    monkeypatch.undo()
    assert CLI.main([*args, "--before", "2016-01-01"]) == 1  # inside the test period
    assert "tuned only on fights before 2015-01-01" in capsys.readouterr().err


_DIAG = {
    "fights": 1,
    "scored_from": "2005-01-01",
    "before": "2015-01-01",
    "glicko": {"c": 1.0, "log_loss": 0.6, "log_loss_c0": 0.6},
    "welo": {"log_loss": 0.6, "log_loss_plain_elo": 0.6},
    "adjusted": {"mse": 0.6, "mse_untuned": 0.6, "k0": 1.0, "rate_min": 0.1},
}


# --- the sets run end to end ---------------------------------------------------------------------


def test_every_feature_set_runs_and_stays_symmetric(world):
    fights, stats, fighters, feats = world
    s = scored_fights(feats, fights)
    for groups in (("ratings",), ("adjusted",), ("context",), ("ratings", "adjusted", "context")):
        pred = walk_forward(s, "2015-01-01", groups=groups, end="2015-12-31")
        assert len(pred) > 20
        for m in ("hgb", "logreg", "blend"):
            assert pred[f"p_{m}"].between(0, 1).all()
