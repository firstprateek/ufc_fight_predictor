import subprocess

import pandas as pd
import pytest
from test_backtest import fake_odds, synthetic
from test_ratings import _stats

import ufc.backtest.ratings_report as RR
from ufc.backtest.protocol import ProtocolError
from ufc.features import build_features


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
        check=True, capture_output=True, text=True,
    ).stdout.strip()  # fmt: skip


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q")
    p = tmp_path / "src/ufc/backtest"
    p.mkdir(parents=True)
    (p / "protocol.py").write_text("FINAL_FEATURE_SET: str | None = None\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "before")
    (p / "protocol.py").write_text('FINAL_FEATURE_SET: str | None = "ratings+adjusted"\n')
    git(tmp_path, "commit", "-qam", "freeze")
    return tmp_path


def test_verify_frozen_checks_the_commit_holds_the_configuration(repo):
    freeze = git(repo, "rev-parse", "HEAD")
    before = git(repo, "rev-parse", "HEAD~1")
    assert RR.verify_frozen(freeze[:10], "ratings+adjusted", repo) == freeze
    with pytest.raises(ProtocolError, match="does not freeze"):
        RR.verify_frozen(before, "ratings+adjusted", repo)
    with pytest.raises(ProtocolError, match="does not freeze"):
        RR.verify_frozen(freeze, "all", repo)
    with pytest.raises(ProtocolError, match="not found"):
        RR.verify_frozen("0123abcd", "ratings+adjusted", repo)
    with pytest.raises(ProtocolError, match="not frozen"):
        RR.verify_frozen(freeze, None, repo)
    # a commit that is not an ancestor of HEAD
    git(repo, "switch", "-q", "-c", "other", "HEAD~1")
    with pytest.raises(ProtocolError, match="not an ancestor"):
        RR.verify_frozen(freeze, "ratings+adjusted", repo)


def test_report_refuses_an_unfrozen_configuration(monkeypatch):
    monkeypatch.setattr(RR, "FINAL_FEATURE_SET", None)
    fights, fighters = synthetic(30)
    feats = build_features(fights, _stats(fights), fighters)
    with pytest.raises(ProtocolError, match="not frozen"):
        RR.ratings_report(
            feats, fights, _stats(fights), pd.DataFrame(), snapshot="s", odds_snapshot="o",
            frozen_commit="HEAD",
        )  # fmt: skip


def test_report_end_to_end_on_synthetic_data(monkeypatch):
    fights, fighters = synthetic(1000)
    stats = _stats(fights)
    feats = build_features(fights, stats, fighters)
    from ufc.backtest.run import scored_fights

    odds = fake_odds(scored_fights(feats, fights), noise=0.3)
    monkeypatch.setattr(RR, "FINAL_FEATURE_SET", "all")
    import ufc.backtest.protocol as P

    monkeypatch.setattr(P, "FINAL_FEATURE_SET", "all")
    monkeypatch.setattr(RR, "verify_frozen", lambda commit, final, repo: "f" * 40)
    monkeypatch.setattr(
        RR,
        "SEGMENTS",
        (
            ("development", "2015-01-01", "2016-06-30"),
            ("holdout", "2016-07-01", None),
            ("full", "2015-01-01", None),
        ),
    )
    # the rule may drop a group on this synthetic data; force the check to see it as "all"
    monkeypatch.setattr(
        RR, "ablation_markdown", lambda preds: ("ABLATION", ("ratings", "adjusted", "context"))
    )
    monkeypatch.setattr(RR, "constants_markdown", lambda f, s: "CONSTANTS")
    monkeypatch.setattr(RR, "run_ablation", lambda scored: {})
    md = RR.ratings_report(
        feats, fights, stats, odds, snapshot="snap", odds_snapshot="osnap", frozen_commit="abc",
        reading="By hand.",
    )  # fmt: skip
    for text in ("snap", "osnap", "f" * 40, "## 1. Development ablation", "ABLATION", "CONSTANTS",
                 "Information test", "Reading (hand-written", "By hand."):  # fmt: skip
        assert text in md
    for seg in ("development", "holdout", "full"):
        assert f"| {seg} | hgb |" in md and f"| {seg} | blend |" in md
    assert "| holdout | all | hgb |" in md and "| development | base | blend |" in md


def test_report_fails_if_the_rule_disagrees_with_the_frozen_configuration(monkeypatch):
    fights, fighters = synthetic(1000)
    stats = _stats(fights)
    feats = build_features(fights, stats, fighters)
    import ufc.backtest.protocol as P

    monkeypatch.setattr(RR, "FINAL_FEATURE_SET", "all")
    monkeypatch.setattr(P, "FINAL_FEATURE_SET", "all")
    monkeypatch.setattr(RR, "verify_frozen", lambda commit, final, repo: "f" * 40)
    monkeypatch.setattr(RR, "ablation_markdown", lambda preds: ("X", ("ratings",)))
    monkeypatch.setattr(RR, "run_ablation", lambda scored: {})
    with pytest.raises(ProtocolError, match="decision rule gives"):
        RR.ratings_report(
            feats, fights, stats, pd.DataFrame(), snapshot="s", odds_snapshot="o", frozen_commit="x"
        )
