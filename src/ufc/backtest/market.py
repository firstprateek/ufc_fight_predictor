"""Scoring against the betting market, and the information test (does the model know anything the
market does not?). Only fights that have odds are used, the same fights for every model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from . import metrics as M
from .models import market, model_names

MARKET = "market"
STACK_START = "2016-01-01"  # 2015 is burn-in for the stacked model
BOOTSTRAP_DRAWS = 1000
PCLIP = 1e-6


def add_market(pred: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    """Add `p_market` (NaN where a fight has no odds) via the `market` model."""
    p1 = pred[["fight_id"]].merge(odds[["fight_id", "p_market_f1"]], on="fight_id", how="left")
    assert (p1.fight_id.to_numpy() == pred.fight_id.to_numpy()).all()
    out = pred.copy()
    out["p_market"] = market(None, None, p1)
    return out


def logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), PCLIP, 1 - PCLIP)
    return np.log(p / (1 - p))


def fit_stack(p_market, p_model, y) -> LogisticRegression:
    """Two inputs (logit market, logit hgb), no intercept, both orientations of every fight, no
    penalty: the answer cannot depend on which fighter is listed first."""
    x = np.column_stack([logit(p_market), logit(p_model)])
    xx = np.vstack([x, -x])
    yy = np.concatenate([np.asarray(y), 1 - np.asarray(y)])
    return LogisticRegression(fit_intercept=False, C=np.inf, max_iter=1000).fit(xx, yy)


def stacked_walk_forward(sub: pd.DataFrame, start: str = STACK_START, model: str = "hgb"):
    """Each quarter from `start`: fit the stack of the market and `model` on out-of-sample
    predictions of earlier quarters (fights with odds), predict the quarter. Returns (rows,
    last_quarter_coefficients, last quarter)."""
    parts, coef, last = [], None, None
    for q in sorted(set(sub.quarter[sub.date >= start])):
        train, test = sub[sub.quarter < q], sub[sub.quarter == q]
        assert train.quarter.max() < q, "stack must train on earlier quarters only"
        fit = fit_stack(train.p_market, train[f"p_{model}"], train.y)
        x = np.column_stack([logit(test.p_market), logit(test[f"p_{model}"])])
        p = fit.predict_proba(x)[:, 1]
        p_swapped = fit.predict_proba(-x)[:, 1]
        assert np.allclose(p + p_swapped, 1.0, atol=1e-9)
        parts.append(test.assign(p_stack=p))
        coef, last = fit.coef_[0], q
    if not parts:
        return sub.iloc[0:0].assign(p_stack=[]), None, None
    return pd.concat(parts), coef, last


def _table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def _fmt(s):
    return [
        s["n"],
        f"{s['log_loss']:.4f}",
        f"{s['brier']:.4f}",
        f"{s['accuracy']:.3f}",
        f"{s['ece']:.4f}",
    ]


def market_subset(pred: pd.DataFrame) -> pd.DataFrame:
    return pred[pred.p_market.notna()].reset_index(drop=True)


def market_summary(sub: pd.DataFrame) -> dict:
    """Overall scores per model (and the market) on the odds fights."""
    return {m: M.summarize(sub[f"p_{m}"], sub.y) for m in [*model_names(sub), MARKET]}


def market_markdown(pred: pd.DataFrame, n_scored: int) -> str:
    sub = market_subset(pred)
    names = [*model_names(sub), MARKET]
    overall = market_summary(sub)
    ev, y = sub.event_id.to_numpy(), sub.y.to_numpy()
    loss = {m: M.point_losses(sub[f"p_{m}"].to_numpy(), y) for m in names}
    md = [
        "## Against the market",
        "",
        f"Only fights with odds are scored here: {len(sub)} of {n_scored} scored fights "
        f"({len(sub) / n_scored:.1%}), the same fights for every model. `market` is the odds-implied "
        "probability with the bookmaker margin removed (pre-fight moneyline, book and timing unknown); "
        "it needs no training.",
        "",
        f"Sanity: the market favourite won {overall[MARKET]['accuracy']:.1%} of these fights "
        "(expected roughly 65%).",
        "",
        "### Overall",
        "",
        _table(
            [[m, *_fmt(overall[m])] for m in names],
            ["model", "fights", "log loss", "Brier", "accuracy", "ECE"],
        ),
        "",
        f"### Log loss minus the market (paired bootstrap over events, {BOOTSTRAP_DRAWS} draws, 95%)",
        "",
        "Positive means worse than the market.",
        "",
    ]
    rows = []
    for m in model_names(sub):
        d, lo, hi = M.paired_bootstrap(loss[m], loss[MARKET], ev, BOOTSTRAP_DRAWS)
        rows.append([m, f"{d:+.4f}", f"[{lo:+.4f}, {hi:+.4f}]"])
    md += [_table(rows, ["model", "difference", "95% interval"]), ""]
    md += ["### Calibration of the market (both fighter views)", ""]
    cal = M.calibration_table(sub.p_market, sub.y)
    md += [
        _table(
            [[f"{lo:.1f}-{hi:.1f}", n, f"{mp:.3f}", f"{obs:.3f}"] for lo, hi, n, mp, obs in cal],
            ["p bin", "fighter views", "mean predicted", "observed win rate"],
        ),
        "",
        "### Per year",
        "",
    ]
    rows = []
    for yr, g in sub.assign(year=sub.date.str[:4]).groupby("year"):
        for m in names:
            rows.append([yr, m, *_fmt(M.summarize(g[f"p_{m}"], g.y))])
    md += [_table(rows, ["year", "model", "fights", "log loss", "Brier", "accuracy", "ECE"]), ""]
    for m in ("hgb", "blend"):
        md += information_test_markdown(sub, m)
    return "\n".join(md)


def information_test_markdown(sub: pd.DataFrame, model: str = "hgb") -> list[str]:
    st, coef, last = stacked_walk_forward(sub, model=model)
    md = [
        f"### Information test: does `{model}` know anything the market does not?",
        "",
        "Each quarter a logistic regression with two inputs, logit of the market probability and logit "
        f"of the `{model}` prediction (no intercept, both orientations, no penalty), is fitted on the "
        "out-of-sample predictions of earlier quarters only (fights with odds; 2015 is burn-in), then "
        f"predicts the quarter. Scored from {STACK_START} on the same fights as the market alone.",
        "",
    ]
    if not len(st):
        return md + ["Not enough fights with odds to run the test.", ""]
    ev, y = st.event_id.to_numpy(), st.y.to_numpy()
    l_stack = M.point_losses(st.p_stack.to_numpy(), y)
    l_mkt = M.point_losses(st.p_market.to_numpy(), y)
    d, lo, hi = M.paired_bootstrap(l_stack, l_mkt, ev, BOOTSTRAP_DRAWS)
    a, b = M.summarize(st.p_market, y), M.summarize(st.p_stack, y)
    md += [
        _table(
            [["market alone", *_fmt(a)], [f"market + {model} stack", *_fmt(b)]],
            ["model", "fights", "log loss", "Brier", "accuracy", "ECE"],
        ),
        "",
        f"Log loss, stack minus market alone: {d:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}] "
        f"(paired bootstrap over events, {BOOTSTRAP_DRAWS} draws). Negative means the model adds "
        "information the market lacks; an interval that includes zero means no evidence that it does.",
        "",
        f"Coefficients of the final quarter's fit ({last}): market logit {coef[0]:.3f}, "
        f"{model} logit {coef[1]:.3f}.",
        "",
    ]
    return md
