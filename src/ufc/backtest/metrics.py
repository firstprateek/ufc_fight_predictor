"""Metrics on p = probability that fighter 1 wins, y = 1 if fighter 1 won."""

from __future__ import annotations

import numpy as np

EPS = 1e-15
BINS = 10


def point_losses(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def log_loss(p, y) -> float:
    return float(point_losses(np.asarray(p, float), np.asarray(y, float)).mean())


def brier(p, y) -> float:
    return float(((np.asarray(p, float) - np.asarray(y, float)) ** 2).mean())


def accuracy(p, y) -> float:
    """Share of fights whose winner has p > 0.5; an exact 0.5 counts as half a hit."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    hit = np.where(p == 0.5, 0.5, ((p > 0.5) == (y == 1)).astype(float))
    return float(hit.mean())


def _both_views(p, y):
    """Every fight seen from both fighters, so calibration covers both sides of 0.5."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    return np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y])


def calibration_table(p, y, bins: int = BINS):
    """Rows (bin_lo, bin_hi, n, mean_predicted, observed_rate) over equal-width bins, both views."""
    pp, yy = _both_views(p, y)
    idx = np.minimum((pp * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append((b / bins, (b + 1) / bins, int(m.sum()), pp[m].mean(), yy[m].mean()))
    return rows


def ece(p, y, bins: int = BINS) -> float:
    """Expected calibration error: bin-weighted mean |observed - predicted| over equal-width bins."""
    rows = calibration_table(p, y, bins)
    total = sum(r[2] for r in rows)
    return float(sum(r[2] * abs(r[4] - r[3]) for r in rows) / total)


def summarize(p, y) -> dict:
    return {
        "n": len(y),
        "log_loss": log_loss(p, y),
        "brier": brier(p, y),
        "accuracy": accuracy(p, y),
        "ece": ece(p, y),
    }


def paired_bootstrap(
    loss_a: np.ndarray, loss_b: np.ndarray, events: np.ndarray, draws: int = 1000, seed: int = 0
):
    """Mean(loss_a - loss_b) with a 95% percentile interval, resampling events with replacement."""
    _, inv = np.unique(events, return_inverse=True)
    k = inv.max() + 1
    diff = np.bincount(inv, weights=loss_a - loss_b, minlength=k)
    cnt = np.bincount(inv, minlength=k).astype(float)
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, k, size=(draws, k))
    est = diff[samples].sum(axis=1) / cnt[samples].sum(axis=1)
    lo, hi = np.percentile(est, [2.5, 97.5])
    return float(diff.sum() / cnt.sum()), float(lo), float(hi)
