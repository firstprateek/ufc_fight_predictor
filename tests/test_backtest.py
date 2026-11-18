import numpy as np
import pandas as pd
import pytest

from ufc.backtest import LeakSuspected, run_backtest
from ufc.backtest import metrics as M
from ufc.backtest.models import MODELS
from ufc.backtest.run import scored_fights, walk_forward
from ufc.features import build_features, swap_orientation


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
