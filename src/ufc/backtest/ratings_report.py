"""`docs/results/ratings.md`: the development ablation, the frozen configuration on development,
holdout and the full period next to `base`, the comparison with the market, and the constants.

Only runs once a commit has frozen the configuration (`FINAL_FEATURE_SET`), and checks that this
commit exists, is an ancestor of HEAD and holds the same configuration, and that the pre-registered
rule still gives it on the development period. Everything is generated except the optional reading
text passed in, which is marked as hand-written.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..features import RATING_PARAMS, parse_feature_set, set_name
from . import market as MK
from . import metrics as M
from .ablation import ablation_markdown, run_ablation
from .protocol import (
    DEV_END,
    FINAL_FEATURE_SET,
    HOLDOUT_START,
    TEST_START,
    ProtocolError,
    check_run,
)
from .run import BOOTSTRAP_DRAWS, scored_fights, walk_forward

SEGMENTS = (
    ("development", TEST_START, DEV_END),
    ("holdout", HOLDOUT_START, None),
    ("full", TEST_START, None),
)
SHOWN = ("hgb", "logreg", "blend")
MARKET_MODELS = ("hgb", "blend")


def verify_frozen(
    commit: str, final: str | None = FINAL_FEATURE_SET, repo: Path = Path(".")
) -> str:
    """The full sha of `commit`, after checking it is an ancestor of HEAD whose protocol.py sets the
    frozen configuration to `final`."""

    def git(*args):
        return subprocess.run(
            ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
        )

    if final is None:
        raise ProtocolError("the configuration is not frozen (FINAL_FEATURE_SET is None)")
    sha = git("rev-parse", "--verify", f"{commit}^{{commit}}")
    if sha.returncode:
        raise ProtocolError(f"frozen commit {commit!r} not found")
    sha = sha.stdout.strip()
    if git("merge-base", "--is-ancestor", sha, "HEAD").returncode:
        raise ProtocolError(f"frozen commit {sha[:10]} is not an ancestor of HEAD")
    src = git("show", f"{sha}:src/ufc/backtest/protocol.py").stdout
    if f'FINAL_FEATURE_SET: str | None = "{final}"' not in src:
        raise ProtocolError(f"frozen commit {sha[:10]} does not freeze {final!r}")
    return sha


def _table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _seg(df: pd.DataFrame, lo: str, hi: str | None) -> pd.DataFrame:
    m = df.date >= lo
    if hi is not None:
        m &= df.date <= hi
    return df[m]


def _ci(a: np.ndarray, b: np.ndarray, events: np.ndarray) -> str:
    d, lo, hi = M.paired_bootstrap(a, b, events, BOOTSTRAP_DRAWS)
    return f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def _loss(df: pd.DataFrame, col: str) -> np.ndarray:
    return M.point_losses(df[col].to_numpy(), df.y.to_numpy())


def final_vs_base_markdown(final: pd.DataFrame, base: pd.DataFrame, name: str) -> str:
    rows = []
    for seg, lo, hi in SEGMENTS:
        f, b = _seg(final, lo, hi), _seg(base, lo, hi)
        for m in SHOWN:
            sf, sb = M.summarize(f[f"p_{m}"], f.y), M.summarize(b[f"p_{m}"], b.y)
            rows.append(
                [
                    seg, m, sf["n"], f"{sb['log_loss']:.4f}", f"{sf['log_loss']:.4f}",
                    f"{sb['accuracy']:.3f}", f"{sf['accuracy']:.3f}",
                    _ci(_loss(f, f"p_{m}"), _loss(b, f"p_{m}"), f.event_id.to_numpy()),
                ]
            )  # fmt: skip
    return _table(
        rows,
        ["period", "model", "fights", "base log loss", f"{name} log loss", "base accuracy",
         f"{name} accuracy", f"{name} minus base, log loss [95%]"],
    )  # fmt: skip


def market_markdown(final: pd.DataFrame, base: pd.DataFrame, name: str) -> tuple[str, str]:
    """(against-the-market table, information-test table), per period, for hgb and blend."""
    rows, info = [], []
    stacks = {}
    for label, pred in ((name, final), ("base", base)):
        sub = MK.market_subset(pred)
        for m in MARKET_MODELS:
            stacks[(label, m)] = MK.stacked_walk_forward(sub, model=m)
    for seg, lo, hi in SEGMENTS:
        for label, pred in ((name, final), ("base", base)):
            sub = _seg(MK.market_subset(pred), lo, hi)
            for m in MARKET_MODELS:
                sm, sk = M.summarize(sub[f"p_{m}"], sub.y), M.summarize(sub.p_market, sub.y)
                rows.append(
                    [
                        seg, label, m, sm["n"], f"{sk['log_loss']:.4f}", f"{sm['log_loss']:.4f}",
                        f"{sk['accuracy']:.3f}", f"{sm['accuracy']:.3f}",
                        _ci(_loss(sub, f"p_{m}"), _loss(sub, "p_market"), sub.event_id.to_numpy()),
                    ]
                )  # fmt: skip
                st, coef, last = stacks[(label, m)]
                st = _seg(st, lo, hi)
                if not len(st):
                    continue
                a, b = M.summarize(st.p_stack, st.y), M.summarize(st.p_market, st.y)
                info.append(
                    [
                        seg, label, m, a["n"], f"{b['log_loss']:.4f}", f"{a['log_loss']:.4f}",
                        _ci(_loss(st, "p_stack"), _loss(st, "p_market"), st.event_id.to_numpy()),
                        f"{coef[0]:.3f} / {coef[1]:.3f} ({last})" if seg == "full" else "",
                    ]
                )  # fmt: skip
    mk = _table(
        rows,
        ["period", "features", "model", "fights with odds", "market log loss", "model log loss",
         "market accuracy", "model accuracy", "model minus market, log loss [95%]"],
    )  # fmt: skip
    it = _table(
        info,
        ["period", "features", "model", "fights", "market alone log loss", "stack log loss",
         "stack minus market [95%]", "final-quarter weights market / model"],
    )  # fmt: skip
    return mk, it


def constants_markdown(fights: pd.DataFrame, stats: pd.DataFrame) -> str:
    from ..features.tune import tune_ratings

    params, diag = tune_ratings(fights, stats)
    same = "reproduces" if params == RATING_PARAMS else "DOES NOT reproduce"
    p = RATING_PARAMS
    g, w, a = diag["glicko"], diag["welo"], diag["adjusted"]
    rows = [
        ["Glicko-1 deviation growth `c`", p.glicko_c, f"log loss {g['log_loss']:.4f} (c = 0: {g['log_loss_c0']:.4f})"],
        ["method-weighted Elo `K`", p.welo_k, f"log loss {w['log_loss']:.4f} (plain Elo K=32: {w['log_loss_plain_elo']:.4f})"],
        ["multiplier, unanimous decision", p.welo_unan, "split / majority decision = 1"],
        ["multiplier, KO/TKO", p.welo_ko, ""],
        ["multiplier, submission", p.welo_sub, ""],
        ["multiplier, other endings", p.welo_other, "DQ, overturned, could not continue, other"],
        ["adjusted `k0`", p.adj_k0, f"standardised mse {a['mse']:.4f} (k0=5, rate_min=0.05: {a['mse_untuned']:.4f})"],
        ["adjusted `rate_min`", p.adj_rate_min, ""],
    ]  # fmt: skip
    return "\n".join(
        [
            f"Tuned on {diag['fights']} fights dated before {diag['before']}, one-step-ahead error "
            f"scored from {diag['scored_from']}. `ufc tune-ratings --before 2015-01-01` {same} "
            "these constants.",
            "",
            _table(rows, ["constant", "value", "one-step-ahead error at the optimum"]),
        ]
    )


def ratings_report(
    features: pd.DataFrame,
    fights: pd.DataFrame,
    stats: pd.DataFrame,
    odds: pd.DataFrame,
    *,
    snapshot: str,
    odds_snapshot: str,
    frozen_commit: str,
    reading: str | None = None,
    repo: Path = Path("."),
) -> str:
    final_set = FINAL_FEATURE_SET
    sha = verify_frozen(frozen_commit, final_set, repo)
    groups = parse_feature_set(final_set)
    name = set_name(groups)
    check_run(final_set, None)
    t0 = time.time()
    scored = scored_fights(features, fights)
    preds = run_ablation(scored)
    ab_md, kept = ablation_markdown(preds)
    if set_name(kept) != name:
        raise ProtocolError(
            f"the decision rule gives {set_name(kept)!r} on development, not the frozen {name!r}"
        )
    base = walk_forward(scored, TEST_START, odds, (), None)
    final = walk_forward(scored, TEST_START, odds, groups, None) if groups else base
    mk_md, info_md = market_markdown(final, base, name)
    elapsed = time.time() - t0
    n = {s: len(_seg(final, lo, hi)) for s, lo, hi in SEGMENTS}
    md = [
        "# Ratings and context features",
        "",
        "Generated by `uv run ufc ratings-report` unless a section says it is hand-written.",
        "",
        f"- data snapshot: `{snapshot}`",
        f"- odds snapshot: `{odds_snapshot}` (`ufc-master.csv`, pre-fight moneyline, book and timing unknown)",
        f"- configuration frozen in commit `{sha}` (feature set `{name}`; the protocol is design principle 8)",
        f"- commands: `uv run ufc fetch && uv run ufc build && uv run ufc features --check && "
        f"uv run ufc tune-ratings && uv run ufc ratings-report --frozen-commit {sha[:10]}`; "
        "the development ablation alone is `uv run ufc backtest --ablation`",
        f"- development = {TEST_START} to {DEV_END} ({n['development']} scored fights), holdout = "
        f"{HOLDOUT_START} to {final.date.max()} ({n['holdout']}), full = both ({n['full']}); "
        "each quarter every model is fitted on all scored fights before it",
        "- hgb and logistic regression hyperparameters are unchanged; `blend` is the mean of their logits",
        f"- walk-forward runs in this report took {elapsed:.0f} s",
        "",
        "## 1. Development ablation",
        "",
        "Made on the development period only, before the configuration was frozen.",
        "",
        ab_md,
        "",
        f"## 2. Final configuration (`{name}`) against `base`",
        "",
        "Same fights for both. Log loss difference is final minus base with a paired-bootstrap 95% "
        f"interval (events resampled, {BOOTSTRAP_DRAWS} draws); negative is better. The holdout was "
        "run once, after the freeze.",
        "",
        final_vs_base_markdown(final, base, name),
        "",
        "## 3. Against the market",
        "",
        "Only fights with odds; the market probability has the bookmaker margin removed. Model minus "
        "market log loss is positive when the model is worse.",
        "",
        mk_md,
        "",
        "### Information test",
        "",
        "Each quarter from 2016-01-01 a logistic regression on the logits of the market and the model "
        "(no intercept, both orientations, no penalty) is fitted on earlier quarters' out-of-sample "
        "predictions and predicts the quarter. The holdout rows use a stack fitted on all earlier "
        "quarters, development included. A negative difference with an interval below zero would mean the "
        "model carries information the market lacks.",
        "",
        info_md,
        "",
        "## 4. Tuned constants",
        "",
        constants_markdown(fights, stats),
        "",
    ]
    if reading:
        md += ["## Reading (hand-written, not generated)", "", reading.strip(), ""]
    return "\n".join(md)
