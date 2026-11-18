import subprocess

import numpy as np
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


def write(repo, text, name="protocol.py"):
    p = repo / "src/ufc/backtest"
    p.mkdir(parents=True, exist_ok=True)
    (p / name).write_text(text)


@pytest.fixture
def repo(tmp_path):
    git(tmp_path, "init", "-q", "-b", "main")
    write(tmp_path, "FINAL_FEATURE_SET: str | None = None\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-qm", "before")
    write(tmp_path, 'FINAL_FEATURE_SET: str | None = "ratings+adjusted"\n')
    git(tmp_path, "commit", "-qam", "freeze")
    return tmp_path


def verify(repo, commit, final="ratings+adjusted"):
    return RR.verify_frozen(commit, final, repo, base_ref="main")


def test_verify_frozen_checks_the_commit_holds_the_configuration(repo):
    freeze = git(repo, "rev-parse", "HEAD")
    before = git(repo, "rev-parse", "HEAD~1")
    assert verify(repo, freeze[:10]) == freeze
    with pytest.raises(ProtocolError, match="freezes None"):
        verify(repo, before)
    with pytest.raises(ProtocolError, match="freezes 'ratings"):
        verify(repo, freeze, "all")
    with pytest.raises(ProtocolError, match="not found"):
        verify(repo, "0123abcd")
    with pytest.raises(ProtocolError, match="not frozen"):
        verify(repo, freeze, None)


def test_verify_frozen_refuses_unmerged_and_non_ancestor_commits(repo):
    git(repo, "switch", "-q", "-c", "feature")
    write(repo, 'FINAL_FEATURE_SET: str | None = "all"\n')
    git(repo, "commit", "-qam", "freeze on a branch")
    unmerged = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ProtocolError, match="not merged into main"):
        verify(repo, unmerged, "all")
    git(repo, "switch", "-q", "main")  # HEAD is main: the branch commit is also not an ancestor
    with pytest.raises(ProtocolError, match="not merged"):
        verify(repo, unmerged, "all")
    old = git(repo, "rev-parse", "HEAD")
    git(repo, "switch", "-q", "-c", "behind", "HEAD~1")  # HEAD behind the freeze
    with pytest.raises(ProtocolError, match="not an ancestor of HEAD"):
        verify(repo, old)


def test_verify_frozen_cannot_be_spoofed_by_a_comment(repo):
    write(repo, '# FINAL_FEATURE_SET: str | None = "all"\nFINAL_FEATURE_SET: str | None = None\n')
    git(repo, "commit", "-qam", "comment only")
    sha = git(repo, "rev-parse", "HEAD")
    with pytest.raises(ProtocolError, match="freezes None"):
        verify(repo, sha, "all")


def test_verify_frozen_requires_src_unchanged_since_the_freeze(repo):
    freeze = git(repo, "rev-parse", "HEAD")
    write(repo, "x = 1\n", "other.py")
    with pytest.raises(ProtocolError, match="uncommitted changes in src"):
        verify(repo, freeze)
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "code after the freeze")
    with pytest.raises(ProtocolError, match="src/ changed after"):
        verify(repo, freeze)
    (repo / "README.md").write_text("docs only\n")  # outside src/: allowed
    git(repo, "revert", "--no-edit", "HEAD")
    git(repo, "add", "-A")
    git(repo, "commit", "-qm", "docs")
    assert verify(repo, freeze) == freeze


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
    monkeypatch.setattr(RR, "verify_frozen", lambda commit, final, repo, base_ref: "f" * 40)
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
    sizes = {
        seg: int(md.split(f"| {seg} | hgb | ")[1].split(" |")[0])
        for seg in ("development", "holdout", "full")
    }
    assert sizes["development"] + sizes["holdout"] == sizes["full"] and sizes["holdout"] > 20
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
    monkeypatch.setattr(RR, "verify_frozen", lambda commit, final, repo, base_ref: "f" * 40)
    monkeypatch.setattr(RR, "ablation_markdown", lambda preds: ("X", ("ratings",)))
    monkeypatch.setattr(RR, "run_ablation", lambda scored: {})
    with pytest.raises(ProtocolError, match="decision rule gives"):
        RR.ratings_report(
            feats, fights, stats, pd.DataFrame(), snapshot="s", odds_snapshot="o", frozen_commit="x"
        )


def test_stack_rows_in_a_segment_do_not_depend_on_later_quarters():
    from ufc.backtest import market as MK
    from ufc.backtest.run import scored_fights, walk_forward

    fights, fighters = synthetic(1000)
    feats = build_features(fights, _stats(fights), fighters)
    scored = scored_fights(feats, fights)
    pred = walk_forward(scored, "2015-01-01", fake_odds(scored, noise=0.3))
    sub = MK.market_subset(pred)
    cut = "2016-06-30"
    full, _, _ = MK.stacked_walk_forward(sub, model="blend")
    early, _, _ = MK.stacked_walk_forward(sub[sub.date <= cut], model="blend")
    a = full[full.date <= cut].set_index("fight_id").p_stack
    b = early.set_index("fight_id").p_stack
    assert len(b) > 50
    np.testing.assert_allclose(a.loc[b.index], b)
    seg = RR._seg(sub, "2016-07-01", None)
    assert seg.date.min() >= "2016-07-01" and len(RR._seg(sub, "2015-01-01", cut)) + len(
        seg
    ) == len(sub)
