import numpy as np
import pandas as pd
import pytest
from test_backtest import synthetic

import ufc.backtest.models as BM
from ufc.backtest import protocol as P
from ufc.backtest.ablation import ablation_markdown, decide, run_ablation
from ufc.backtest.models import blend, models_for
from ufc.backtest.run import run_backtest, scored_fights, walk_forward
from ufc.features import (
    DIMENSIONS,
    UnknownFeatureSet,
    build_features,
    check_columns,
    column_group,
    feature_columns,
    parse_feature_set,
    set_name,
)


def with_group_columns(feats, seed=5):
    """Add fake columns for every group (the real ones come from the feature builders)."""
    rng = np.random.default_rng(seed)
    out = feats.copy()
    names = ["glicko_r", "welo", *[f"adj_{d}_exp" for d in DIMENSIONS], "ctx_sos_mean"]
    for n in names:
        a = rng.normal(0, 1, len(out))
        out[f"f1_{n}"], out[f"f2_{n}"] = a, rng.normal(0, 1, len(out))
        out[f"diff_{n}"] = out[f"f1_{n}"] - out[f"f2_{n}"]
    return out


@pytest.fixture(scope="module")
def data():
    fights, fighters = synthetic(1100)
    stats = pd.DataFrame(columns=["fight_id", "fighter_id", "round"])
    feats = with_group_columns(build_features(fights, stats, fighters))
    return fights, feats


def test_column_groups_and_feature_set_names():
    assert column_group("f1_glicko_rd") == "ratings" and column_group("diff_welo") == "ratings"
    assert column_group("f2_adj_slpm_off") == "adjusted" and column_group("diff_ctx_x") == "context"
    assert column_group("f1_elo") == "base" and column_group("diff_slpm_career") == "base"
    assert parse_feature_set("base") == () and parse_feature_set("all") == (
        "ratings",
        "adjusted",
        "context",
    )
    assert parse_feature_set("context+ratings") == ("ratings", "context")
    assert set_name(parse_feature_set("adjusted+ratings")) == "ratings+adjusted"
    assert set_name(()) == "base" and set_name(parse_feature_set("all")) == "all"
    with pytest.raises(UnknownFeatureSet):
        parse_feature_set("everything")


def test_feature_columns_by_set(data):
    _, feats = data
    base = feature_columns(feats, ())
    assert not any(column_group(c) != "base" for c in base)
    r = feature_columns(feats, ("ratings",))
    assert set(r) - set(base) == {
        f"{p}_{n}" for p in ("f1", "f2", "diff") for n in ("glicko_r", "welo")
    }
    assert set(feature_columns(feats, ("ratings", "adjusted", "context"))) == set(
        feature_columns(feats)
    )


def test_check_columns_refuses_a_set_without_its_columns(data):
    _, feats = data
    check_columns(feats, ("ratings", "adjusted", "context"))
    plain = feats[[c for c in feats.columns if column_group(c) == "base"]]
    with pytest.raises(UnknownFeatureSet, match="ratings"):
        check_columns(plain, ("ratings",))
    with pytest.raises(UnknownFeatureSet, match="adjusted"):
        check_columns(feats.drop(columns="diff_adj_kd15_exp"), ("adjusted",))


def test_protocol_refuses_unfrozen_sets_past_development():
    P.check_run("base", None, final=None)
    P.check_run("base", "2025-01-01", final=None)
    P.check_run("all", P.DEV_END, final=None)
    P.check_run("all", "2019-01-01", final=None)
    for end in (None, "2023-01-01", "2026-01-01"):
        with pytest.raises(P.ProtocolError, match="not frozen"):
            P.check_run("all", end, final=None)
    P.check_run("ratings+adjusted", None, final="adjusted+ratings")
    with pytest.raises(P.ProtocolError, match="only the frozen"):
        P.check_run("all", None, final="ratings+adjusted")
    with pytest.raises(P.ProtocolError, match="YYYY-MM-DD"):
        P.check_run("base", "2022/12/31")


def test_run_backtest_enforces_the_protocol(data, tmp_path):
    fights, feats = data
    with pytest.raises(P.ProtocolError):
        run_backtest(feats, fights, "2015-01-01", tmp_path / "r", features_set="all")
    assert not (tmp_path / "r").exists()


def test_walk_forward_end_drops_later_fights_and_ignores_them(data):
    fights, feats = data
    s = scored_fights(feats, fights)
    end = "2015-12-31"
    a = walk_forward(s, "2015-01-01", end=end)
    assert a.date.max() <= end and a.date.min() >= "2015-01-01"
    flipped = s.copy()
    later = flipped.date > end
    assert later.sum() > 50
    flipped.loc[later, "y"] = 1 - flipped.loc[later, "y"]
    flipped.loc[later, "diff_elo"] = 123.0  # features after the end cannot matter either
    b = walk_forward(flipped, "2015-01-01", end=end)
    cols = [c for c in a.columns if c.startswith("p_")]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)
    # the same fights get the same predictions with or without an end
    full = walk_forward(s, "2015-01-01")
    pd.testing.assert_frame_equal(
        full[full.date <= end].reset_index(drop=True)[cols], a[cols], check_exact=True
    )


def test_blend_is_the_mean_of_logits_and_symmetric():
    p, q = np.array([0.7, 0.2, 0.5]), np.array([0.6, 0.4, 0.9])
    b = blend(p, q)
    lg = lambda x: np.log(x / (1 - x))  # noqa: E731
    np.testing.assert_allclose(lg(b), (lg(p) + lg(q)) / 2)
    np.testing.assert_allclose(blend(1 - p, 1 - q), 1 - b)


def test_walk_forward_has_blend_between_its_parts(data):
    fights, feats = data
    pred = walk_forward(scored_fights(feats, fights), "2016-01-01", end="2016-12-31")
    np.testing.assert_allclose(pred.p_blend, blend(pred.p_hgb, pred.p_logreg))


def test_models_for_a_set_use_its_columns_only(data, monkeypatch):
    fights, feats = data
    s = scored_fights(feats, fights)
    train, test = s[s.date < "2015-01-01"], s[s.date >= "2015-01-01"].head(30)
    seen = {}
    real = BM.HistGradientBoostingClassifier.fit

    def spy(self, X, y, *a, **k):
        seen["hgb"] = X.shape[1]
        return real(self, X, y, *a, **k)

    monkeypatch.setattr(BM.HistGradientBoostingClassifier, "fit", spy)
    for groups in ((), ("ratings",), ("adjusted", "context")):
        models_for(groups)["hgb"](train, train.y.to_numpy(), test)
        want = [c for c in feature_columns(train, groups) if train[c].astype(float).nunique() >= 2]
        assert seen["hgb"] == len(want)
    assert BM.logreg_inputs(()) == BM.LOGREG_DIFFS
    assert BM.logreg_inputs(("ratings",))[-2:] == ["diff_glicko_r", "diff_welo"]
    assert BM.logreg_inputs(("adjusted",))[-5:] == [f"diff_adj_{d}_exp" for d in DIMENSIONS]
    assert BM.logreg_inputs(("context",)) == BM.LOGREG_DIFFS
    for groups in (("ratings",), ("adjusted",), ("ratings", "adjusted", "context")):
        for name in ("logreg", "hgb"):
            m = models_for(groups)[name]
            p, q = (
                m(train, train.y.to_numpy(), test),
                m(train, train.y.to_numpy(), BM.swap_orientation(test)),
            )
            np.testing.assert_allclose(p + q, 1.0, atol=1e-9)


def test_labels_of_flagged_fights_never_train_a_larger_set():
    fights, fighters = synthetic(1000)
    fighters.loc[fighters.fighter_id.isin(["f00", "f01", "f02"]), "reach_cm"] = None
    stats = pd.DataFrame(columns=["fight_id", "fighter_id", "round"])
    feats = with_group_columns(build_features(fights, stats, fighters))
    s = scored_fights(feats, fights)
    flipped = s.copy()
    hit = BM.flagged(flipped) & (flipped.date < "2016-01-01")
    assert hit.sum() > 10
    flipped.loc[hit, "y"] = 1 - flipped.loc[hit, "y"]
    g = ("ratings", "adjusted", "context")
    a = walk_forward(s, "2015-01-01", groups=g, end="2016-12-31")
    b = walk_forward(flipped, "2015-01-01", groups=g, end="2016-12-31")
    cols = [c for c in a.columns if c.startswith("p_")]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)


def test_ablation_runs_every_set_on_the_same_development_fights(data):
    fights, feats = data
    s = scored_fights(feats, fights)
    preds = run_ablation(
        s, sets=("base", "ratings", "adjusted", "context", "all"), end="2016-12-31"
    )
    assert set(preds) == {"base", "ratings", "adjusted", "context", "all"}
    ids = preds["base"].fight_id
    assert all(p.fight_id.equals(ids) for p in preds.values())
    md, kept = ablation_markdown(preds)
    assert "(reference)" in md and "Decision rule" in md and "Final configuration" in md
    assert all(f"| {m} | all |" in md for m in ("hgb", "logreg", "blend"))
    with pytest.raises(ValueError, match="development"):
        run_ablation(s, end="2023-06-30")


def fake_preds(worse: dict[str, float], n=600, seed=0):
    """Predictions for the five sets: base is a noisy signal; a set is `worse` by shrinking it."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    base = 1 / (1 + np.exp(-(0.6 * (2 * y - 1) + rng.normal(0, 1, n))))
    events = np.arange(n) // 4
    out = {}
    for name in ("base", "ratings", "adjusted", "context", "all"):
        p = base.copy()
        bad = worse.get(name, 0.0)
        if bad:  # destroy information: push towards a wrong, confident answer
            p = np.clip(base - bad * (2 * y - 1) * 0.3, 0.02, 0.98)
        out[name] = pd.DataFrame(
            {"fight_id": np.arange(n), "event_id": events, "y": y, "date": "2016-01-01"}
        ).assign(p_hgb=p, p_logreg=p, p_blend=p)
    return out


def test_decision_rule_drops_only_a_group_clearly_worse_than_base():
    kept, detail = decide(fake_preds({"context": 1.0}))
    assert kept == ("ratings", "adjusted") and detail["context"][3] and not detail["ratings"][3]
    assert decide(fake_preds({}))[0] == ("ratings", "adjusted", "context")  # ties are kept
    # a group that is better than base is never dropped
    better = fake_preds({})
    better["adjusted"] = better["adjusted"].assign(p_hgb=lambda d: np.where(d.y == 1, 0.8, 0.2))
    assert "adjusted" in decide(better)[0]


def test_cli_refuses_an_unfrozen_set_past_development(tmp_path, capsys):
    from pathlib import Path

    from ufc.build import build
    from ufc.cli import main

    fix = Path(__file__).parent / "fixtures"
    build(fix / "source", tmp_path / "processed", fix / "overrides")
    assert main(["--data-dir", str(tmp_path), "features"]) == 0
    capsys.readouterr()
    args = ["--data-dir", str(tmp_path), "backtest", "--reports-dir", str(tmp_path / "rep")]
    assert main([*args, "--features-set", "all"]) == 1
    assert "not frozen" in capsys.readouterr().err
    # on the development period the set is allowed, but the columns must exist
    assert main([*args, "--features-set", "ratings", "--end", "2022-12-31"]) == 1
    assert "rebuild with `ufc features`" in capsys.readouterr().err
    assert main([*args, "--features-set", "nonsense", "--end", "2022-12-31"]) == 1
    assert "unknown feature set" in capsys.readouterr().err
