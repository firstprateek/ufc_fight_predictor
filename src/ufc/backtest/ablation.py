"""Development ablation and the pre-registered decision rule.

Every feature set runs the same walk-forward on the development period only. The final
configuration is `all`, minus any group whose development log loss for `hgb` is worse than the
`base` set's, with a paired-bootstrap 95% interval (events resampled) that excludes zero. No other
selection is made.
"""

from __future__ import annotations

import pandas as pd

from ..features import BASE, GROUPS, check_columns, parse_feature_set, set_name
from . import metrics as M
from .protocol import ABLATION_SETS, DEV_END, TEST_START
from .run import BOOTSTRAP_DRAWS, walk_forward

MODELS_SHOWN = ("hgb", "logreg", "blend")
DECISION_MODEL = "hgb"


def run_ablation(scored: pd.DataFrame, sets=ABLATION_SETS, end: str = DEV_END) -> dict:
    """set name -> predictions, every set on the same development walk-forward."""
    if end > DEV_END:
        raise ValueError(f"the ablation is a development-period run (end <= {DEV_END}), got {end}")
    out = {}
    for name in sets:
        groups = parse_feature_set(name)
        check_columns(scored, groups)
        out[name] = walk_forward(scored, TEST_START, None, groups, end)
    first = out[next(iter(out))].fight_id.to_numpy()
    assert all((p.fight_id.to_numpy() == first).all() for p in out.values())
    return out


def _delta(preds: dict, name: str, model: str):
    p, b = preds[name], preds[BASE]
    la = M.point_losses(p[f"p_{model}"].to_numpy(), p.y.to_numpy())
    lb = M.point_losses(b[f"p_{model}"].to_numpy(), b.y.to_numpy())
    return M.paired_bootstrap(la, lb, p.event_id.to_numpy(), BOOTSTRAP_DRAWS)


def decide(preds: dict) -> tuple[tuple[str, ...], dict]:
    """The pre-registered rule. Returns (groups kept, {group: (diff, lo, hi, dropped)})."""
    detail, kept = {}, []
    for g in GROUPS:
        d, lo, hi = _delta(preds, g, DECISION_MODEL)
        dropped = bool(lo > 0)  # worse than base, interval excludes zero
        detail[g] = (d, lo, hi, dropped)
        if not dropped:
            kept.append(g)
    return tuple(kept), detail


def ablation_markdown(preds: dict) -> tuple[str, tuple[str, ...]]:
    base = preds[BASE]
    rows = []
    for model in MODELS_SHOWN:
        for name, p in preds.items():
            s = M.summarize(p[f"p_{model}"], p.y)
            if name == BASE:
                diff = "(reference)"
            else:
                d, lo, hi = _delta(preds, name, model)
                diff = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            rows.append([model, name, s["n"], f"{s['log_loss']:.4f}", f"{s['accuracy']:.3f}", diff])
    kept, detail = decide(preds)
    final = set_name(kept)
    md = [
        f"Development period {base.date.min()} to {base.date.max()}, {len(base)} scored fights. "
        "Log loss difference is the set minus `base`, with a paired-bootstrap 95% interval "
        f"(events resampled, {BOOTSTRAP_DRAWS} draws); negative is better.",
        "",
        _table(rows, ["model", "feature set", "fights", "log loss", "accuracy", "minus base"]),
        "",
        f"Decision rule (`{DECISION_MODEL}`, each group against `base`; dropped if worse with an "
        "interval above zero):",
        "",
        _table(
            [
                [g, f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]", "dropped" if drop else "kept"]
                for g, (d, lo, hi, drop) in detail.items()
            ],
            ["group", "log loss minus base", "outcome"],
        ),
        "",
        f"Final configuration: `{final}`.",
    ]
    return "\n".join(md), kept


def _table(rows, header):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


__all__ = ["ablation_markdown", "decide", "run_ablation"]
