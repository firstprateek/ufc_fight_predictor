import numpy as np
import pandas as pd
import pytest

import ufc.backtest.models as BM
from ufc.backtest import LeakSuspected, run_backtest
from ufc.backtest import metrics as M
from ufc.backtest.models import MODELS, flagged
from ufc.backtest.run import scored_fights, walk_forward
from ufc.features import build_features, feature_columns, swap_orientation


def synthetic(n_fights=700, seed=1):
    """Fighters with a hidden skill; 2012-2017 events a week apart, three bouts each."""
    rng = np.random.default_rng(seed)
    ids = [f"f{i:02d}" for i in range(24)]
    skill = dict(zip(ids, rng.normal(0, 1, len(ids)), strict=True))
    fighters = pd.DataFrame(
        {
            "fighter_id": ids,
            "height_cm": rng.normal(178, 7, len(ids)),
            "reach_cm": rng.normal(183, 8, len(ids)),
            "stance": rng.choice(["Orthodox", "Southpaw"], len(ids)),
            "dob": [f"{1980 + i % 12}-03-01" for i in range(len(ids))],
        }
    )
    rows, day = [], pd.Timestamp("2012-01-07")
    while len(rows) < n_fights:
        a = rng.permutation(ids)
        for b in range(3):
            f1, f2 = sorted((a[2 * b], a[2 * b + 1]))
            p = 1 / (1 + np.exp(-(skill[f1] - skill[f2])))
            win = f1 if rng.random() < p else f2
            rows.append(
                {
                    "fight_id": f"x{len(rows):05d}", "event_id": f"e{day.date()}",
                    "date": str(day.date()), "bout_order": b + 1, "fighter_1_id": f1,
                    "fighter_2_id": f2, "winner_id": win, "outcome": "win",
                    "method": "KO/TKO", "end_round": 1, "end_time_s": 100,
                    "weight_class": "Lightweight", "gender": "male",
                }
            )  # fmt: skip
        day += pd.Timedelta(days=7)
    fights = pd.DataFrame(rows).astype({"end_round": "Int64", "end_time_s": "Int64"})
    return fights, fighters


@pytest.fixture(scope="module")
def data():
    fights, fighters = synthetic()
    stats = pd.DataFrame(columns=["fight_id", "fighter_id", "round"])
    return fights, build_features(fights, stats, fighters)


def test_metrics_known_values():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.8, 0.2, 0.5, 0.5])
    assert M.log_loss(p, y) == pytest.approx(-(2 * np.log(0.8) + 2 * np.log(0.5)) / 4)
    assert M.brier(p, y) == pytest.approx((0.04 + 0.04 + 0.25 + 0.25) / 4)
    assert M.accuracy(p, y) == pytest.approx((1 + 1 + 0.5 + 0.5) / 4)
    assert M.log_loss(np.array([1.0]), np.array([0])) < 40  # clipped, finite


def test_ece_perfect_and_overconfident():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 20000)
    assert M.ece(p, (rng.uniform(0, 1, p.size) < p).astype(float)) < 0.02
    assert M.ece(np.full(100, 0.9), np.array([1.0] * 50 + [0.0] * 50)) > 0.3


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(0)
    ev = np.repeat(np.arange(100), 5)
    a, b = rng.uniform(0.5, 0.6, 500), rng.uniform(0.6, 0.7, 500)
    d, lo, hi = M.paired_bootstrap(a, b, ev, draws=200)
    assert lo <= d <= hi < 0
    assert M.paired_bootstrap(a, b, ev, draws=200) == (d, lo, hi)  # seeded


def test_scored_fights_drop_draws_and_label_fighter_1():
    fights, _ = synthetic(30)
    fights.loc[0, ["outcome", "winner_id"]] = ["draw", None]
    feats = build_features(
        fights, pd.DataFrame(columns=["fight_id", "fighter_id", "round"]), synthetic(30)[1]
    )
    s = scored_fights(feats, fights)
    assert len(s) == 29 and "x00000" not in set(s.fight_id)
    w = fights.set_index("fight_id").winner_id
    assert (s.y == (s.fight_id.map(w) == s.fighter_1_id)).all()


def test_walk_forward_trains_only_on_the_past(data, monkeypatch):
    fights, feats = data
    seen = []
    real = MODELS["logreg"]

    def spy(train, y, test):
        seen.append((train.date.max(), test.date.min(), len(train)))
        return real(train, y, test)

    monkeypatch.setitem(MODELS, "logreg", spy)
    pred = walk_forward(scored_fights(feats, fights), "2015-01-01")
    assert seen and all(tmax < tmin for tmax, tmin, _ in seen)
    assert pred.date.min() >= "2015-01-01"
    assert all(
        pred.date[pred.quarter == q].between(q, q[:4] + "-12-31").all() for q in pred.quarter
    )
    assert pred.fight_id.is_unique
    for m in MODELS:
        assert pred[f"p_{m}"].between(0, 1).all()


def test_models_are_orientation_symmetric(data):
    fights, feats = data
    s = scored_fights(feats, fights)
    train, test = s[s.date < "2015-01-01"], s[s.date >= "2016-06-01"].head(40)
    for name in ("elo", "logreg", "hgb"):
        p = MODELS[name](train, train.y.to_numpy(), test)
        q = MODELS[name](train, train.y.to_numpy(), swap_orientation(test))
        np.testing.assert_allclose(p + q, 1.0, atol=1e-9)


def test_run_writes_report_and_learns_something(data, tmp_path):
    fights, feats = data
    overall = run_backtest(
        feats, fights, "2015-01-01", tmp_path / "r", "abc", "uv run ufc backtest"
    )
    assert overall["elo"]["log_loss"] < overall["coin_flip"]["log_loss"]
    text = (tmp_path / "r" / "report.md").read_text()
    assert "snapshot: `abc`" in text and "paired bootstrap" in text and "Per year" in text
    assert len(pd.read_parquet(tmp_path / "r" / "predictions.parquet")) > 100


def test_tripwire_fires_on_a_leak(data, tmp_path):
    fights, feats = data
    s = scored_fights(feats, fights)
    leaky = feats.merge(s[["fight_id", "y"]], on="fight_id", how="left")
    leaky = leaky.assign(diff_leak=leaky.y.fillna(0) * 2 - 1).drop(columns="y")  # the answer
    with pytest.raises(LeakSuspected, match="leak is likely"):
        run_backtest(leaky, fights, "2015-01-01", tmp_path / "r")
    assert (tmp_path / "r" / "report.md").exists()  # the report is still written


def test_labels_from_the_test_quarter_on_never_reach_earlier_predictions(data):
    fights, feats = data
    s = scored_fights(feats, fights)
    base = walk_forward(s, "2015-01-01")
    cut = "2015-10-01"
    flipped = s.copy()
    flipped.loc[flipped.date >= cut, "y"] = 1 - flipped.loc[flipped.date >= cut, "y"]
    alt = walk_forward(flipped, "2015-01-01")
    early = base.date < cut
    cols = [f"p_{m}" for m in MODELS]
    pd.testing.assert_frame_equal(base.loc[early, cols], alt.loc[early, cols])
    assert not np.allclose(base.loc[~early, "p_hgb"], alt.loc[~early, "p_hgb"])


def test_swap_is_an_involution(data):
    _, feats = data
    pd.testing.assert_frame_equal(swap_orientation(swap_orientation(feats)), feats)


def test_paired_bootstrap_interval_contains_zero_under_the_null():
    rng = np.random.default_rng(1)
    ev = np.repeat(np.arange(150), 4)
    a, b = rng.uniform(0.5, 0.8, 600), rng.uniform(0.5, 0.8, 600)
    d, lo, hi = M.paired_bootstrap(a, b, ev, draws=300)
    assert lo < 0 < hi


@pytest.fixture(scope="module")
def flagged_data():
    """Like `data`, but six fighters lack reach (so their fights carry a snapshot flag)."""
    fights, fighters = synthetic()
    fighters.loc[fighters.fighter_id.isin([f"f{i:02d}" for i in range(6)]), "reach_cm"] = None
    stats = pd.DataFrame(columns=["fight_id", "fighter_id", "round"])
    return fights, build_features(fights, stats, fighters)


def test_flags_never_reach_a_feature_matrix_and_flagged_fights_are_not_trained_on(
    flagged_data, monkeypatch
):
    fights, feats = flagged_data
    assert feats.f1_snapshot_imputed.dtype == bool and flagged(feats).any()
    assert not any("snapshot_imputed" in c for c in feature_columns(feats))
    mats = []
    real_fit = BM.HistGradientBoostingClassifier.fit

    def spy(self, X, y, *a, **k):
        mats.append(X)
        return real_fit(self, X, y, *a, **k)

    monkeypatch.setattr(BM.HistGradientBoostingClassifier, "fit", spy)
    s = scored_fights(feats, fights)
    train, test = s[s.date < "2015-01-01"], s[s.date >= "2015-01-01"].head(30)
    assert flagged(train).any()
    MODELS["hgb"](train, train.y.to_numpy(), test)
    cols = [c for c in feature_columns(train, ()) if train[c].astype(float).nunique() >= 2]
    assert mats[0].shape == (2 * (~flagged(train)).sum(), len(cols))


def test_models_refuse_flagged_training_rows_and_flag_columns(flagged_data):
    fights, feats = flagged_data
    s = scored_fights(feats, fights)
    train = s[s.date < "2015-01-01"]
    assert flagged(train).any()
    with pytest.raises(LeakSuspected):
        BM._symmetric(lambda: None, lambda d: d)(train, train.y.to_numpy(), train)
    with pytest.raises(LeakSuspected, match="snapshot_imputed"):
        BM._guard_columns(["diff_elo", "f1_snapshot_imputed"])
    with pytest.raises(LeakSuspected, match="snapshot_imputed"):
        BM._logreg_matrix(train.assign(diff_snapshot_imputed=1.0), extra=["diff_snapshot_imputed"])


def test_labels_of_flagged_fights_have_no_influence(flagged_data):
    fights, feats = flagged_data
    s = scored_fights(feats, fights)
    base = walk_forward(s, "2015-01-01")
    flipped = s.copy()
    hit = flagged(flipped) & (flipped.date < "2017-01-01")
    assert hit.sum() > 20
    flipped.loc[hit, "y"] = 1 - flipped.loc[hit, "y"]
    alt = walk_forward(flipped, "2015-01-01")
    cols = [f"p_{m}" for m in MODELS]
    pd.testing.assert_frame_equal(base[cols], alt[cols], check_exact=True)


def test_report_counts_flagged_fights(flagged_data, tmp_path):
    fights, feats = flagged_data
    run_backtest(feats, fights, "2015-01-01", tmp_path / "r")
    text = (tmp_path / "r" / "report.md").read_text()
    assert "excluded from training, all history" in text
    assert "last training fold" in text and "carrying a flag" in text
    assert "| hgb | flagged |" in text and "| logreg | unflagged |" in text


# --- market benchmark ---------------------------------------------------------------------------


def fake_odds(s, noise=0.5, seed=3):
    """Odds for every scored fight but every 5th: the true win probability seen through noise."""
    rng = np.random.default_rng(seed)
    p = 1 / (1 + 10 ** (-s.diff_elo.to_numpy() / 400))
    z = np.log(p / (1 - p)) + rng.normal(0, noise, len(s))
    odds = pd.DataFrame({"fight_id": s.fight_id, "p_market_f1": 1 / (1 + np.exp(-z))})
    return odds.iloc[np.arange(len(odds)) % 5 != 0].reset_index(drop=True)


def test_market_model_is_the_odds_probability_and_orientation_symmetric():
    from ufc.backtest.models import market

    t = pd.DataFrame({"p_market_f1": [0.7, 0.2, np.nan]})
    np.testing.assert_allclose(market(None, None, t), [0.7, 0.2, np.nan])
    flipped = t.assign(p_market_f1=1 - t.p_market_f1)
    np.testing.assert_allclose(
        market(None, None, t) + market(None, None, flipped), [1.0, 1.0, np.nan]
    )
    assert "market" not in MODELS  # the all-fights tables are unchanged


def test_market_column_is_scored_only_with_odds_and_never_reaches_models(data):
    fights, feats = data
    s = scored_fights(feats, fights)
    odds = fake_odds(s)
    base = walk_forward(s, "2015-01-01")
    pred = walk_forward(s, "2015-01-01", odds)
    cols = [f"p_{m}" for m in MODELS]
    pd.testing.assert_frame_equal(base[cols], pred[cols], check_exact=True)  # models untouched
    assert pred.p_market.notna().sum() == pred.fight_id.isin(odds.fight_id).sum() > 100
    assert not any("market" in c for c in feature_columns(feats))
    m = pred.set_index("fight_id").p_market.dropna()
    np.testing.assert_allclose(m, odds.set_index("fight_id").p_market_f1.loc[m.index])


def test_stack_trains_only_on_earlier_quarters_and_is_symmetric():
    from ufc.backtest import market as MK

    rng = np.random.default_rng(0)
    n = 900
    q = np.repeat(pd.period_range("2015Q1", periods=9, freq="Q").astype(str), n // 9)
    pm = rng.uniform(0.2, 0.8, n)
    pred = pd.DataFrame(
        {
            "quarter": [f"{x[:4]}-{(int(x[-1]) - 1) * 3 + 1:02d}-01" for x in q],
            "p_market": pm,
            "p_hgb": np.clip(pm + rng.normal(0, 0.1, n), 0.05, 0.95),
            "y": (rng.uniform(size=n) < pm).astype(int),
            "event_id": np.arange(n) // 3,
            "fight_id": np.arange(n),
        }
    )
    pred["date"] = pred.quarter
    st, coef, last = MK.stacked_walk_forward(pred)
    assert st.date.min() >= MK.STACK_START and last == "2017-01-01" and len(coef) == 2
    # market is calibrated here, so the stack should lean on it, not on the noisy copy
    assert coef[0] > coef[1]
    # a future label changes nothing earlier
    flipped = pred.copy()
    flipped.loc[flipped.quarter >= "2016-07-01", "y"] = 1 - flipped.y
    st2, _, _ = MK.stacked_walk_forward(flipped)
    early = st.quarter < "2016-07-01"
    np.testing.assert_allclose(st.p_stack[early], st2.p_stack[st2.quarter < "2016-07-01"])


def test_run_writes_market_sections(data, tmp_path):
    fights, feats = data
    odds = fake_odds(scored_fights(feats, fights))
    overall = run_backtest(
        feats, fights, "2015-01-01", tmp_path / "r", "abc", "uv run ufc backtest", odds, "osha"
    )
    assert overall["market"]["accuracy"] < 0.75
    text = (tmp_path / "r" / "report.md").read_text()
    assert "Against the market" in text and "Information test" in text
    assert "minus the market" in text and "Calibration of the market" in text
    mk = (tmp_path / "r" / "market.md").read_text()
    assert "odds snapshot: `osha`" in mk and "data snapshot: `abc`" in mk and "Coefficients" in mk
    # the existing all-fights section is unchanged in shape
    assert "### Log loss minus Elo" in text and "| market |" in text


def test_market_tripwire_fires_on_odds_that_know_the_result(data, tmp_path):
    fights, feats = data
    s = scored_fights(feats, fights)
    leak = pd.DataFrame({"fight_id": s.fight_id, "p_market_f1": np.where(s.y == 1, 0.9, 0.1)})
    with pytest.raises(LeakSuspected, match="market"):
        run_backtest(feats, fights, "2015-01-01", tmp_path / "r", odds=leak)
