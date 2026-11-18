"""The four models. Hyperparameters are fixed here and never tuned against the test period.

Each model is a function (train, y_train, test) -> p, the probability that fighter 1 wins.
Fitted models are trained on both orientations of every training fight and predict
p = (p(f1, f2) + 1 - p(f2, f1)) / 2, so the answer cannot depend on which fighter is listed first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ..features import FLAG, LOGREG_EXTRA, feature_columns, swap_orientation


class LeakSuspected(Exception):
    pass


F1_FLAG, F2_FLAG = f"f1_{FLAG}", f"f2_{FLAG}"


def flagged(df: pd.DataFrame) -> np.ndarray:
    """Fights where either fighter had a snapshot field filled in."""
    return (df[F1_FLAG] | df[F2_FLAG]).to_numpy(dtype=bool)


def _training_set(train: pd.DataFrame, y: np.ndarray):
    """Drop every fight with a filled-in snapshot value: a model never sees a label next to one."""
    keep = ~flagged(train)
    return train[keep], np.asarray(y)[keep]


def _guard_columns(cols) -> None:
    bad = [c for c in cols if FLAG in c]
    if bad:
        raise LeakSuspected(f"snapshot_imputed columns reached a model: {bad}")


LOGREG_DIFFS = [
    "diff_elo",
    "diff_age",
    "diff_reach_cm",
    "diff_n_fights",
    "diff_win_rate",
    "diff_sig_diff",  # career significant strikes landed minus absorbed per minute
]

# Chosen a priori (common conservative defaults for ~10k rows), not tuned on any data here.
HGB_PARAMS = dict(
    max_iter=200,
    learning_rate=0.05,
    max_depth=3,
    min_samples_leaf=40,
    l2_regularization=1.0,
    early_stopping=False,
    random_state=0,
)
LOGREG_C = 1.0


def coin_flip(train, y, test) -> np.ndarray:
    return np.full(len(test), 0.5)


def elo_expected(train, y, test) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-test["diff_elo"].to_numpy(dtype=float) / 400.0))


def logreg_inputs(groups: tuple[str, ...] = ()) -> list[str]:
    """Logistic-regression columns: the base inputs plus the extras of every group that is on."""
    return [*LOGREG_DIFFS, *(c for g in groups for c in LOGREG_EXTRA[g])]


def _logreg_matrix(df: pd.DataFrame, extra=()) -> np.ndarray:
    cols = [*LOGREG_DIFFS, *extra]
    _guard_columns(cols)
    d = df.copy()
    d["diff_sig_diff"] = d["diff_slpm_career"].astype(float) - d["diff_sapm_career"].astype(float)
    return d[cols].to_numpy(dtype=float)


def _symmetric(make, matrix):
    """Wrap a classifier factory in the two-orientation train / averaged predict protocol."""

    def run(train, y, test) -> np.ndarray:
        if flagged(train).any():
            raise LeakSuspected("a training row has a filled-in snapshot value")
        both = pd.concat([train, swap_orientation(train)], ignore_index=True)
        labels = np.concatenate([y, 1 - y])
        model = make().fit(matrix(both), labels)

        def raw(df):
            return model.predict_proba(matrix(df))[:, 1]

        return (raw(test) + 1.0 - raw(swap_orientation(test))) / 2.0

    return run


def make_logistic_regression(groups: tuple[str, ...] = ()):
    """Logistic regression on its base inputs plus the extras of the feature groups that are on."""
    extra = logreg_inputs(groups)[len(LOGREG_DIFFS) :]
    return lambda train, y, test: _logistic_regression(train, y, test, extra)


def logistic_regression(train, y, test) -> np.ndarray:
    return _logistic_regression(train, y, test, ())


def _logistic_regression(train, y, test, extra) -> np.ndarray:
    train, y = _training_set(train, y)

    def make():
        return make_pipeline(
            # train medians + missing flags. Only history columns can be null (debutants, real
            # point-in-time information); the physical columns never are, so no flag can encode
            # snapshot missingness.
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(C=LOGREG_C, max_iter=1000),
        )

    return _symmetric(make, lambda d: _logreg_matrix(d, extra))(train, y, test)


def make_gradient_boosting(groups: tuple[str, ...] = ()):
    """`hgb` on every numeric column of the feature set (base plus the groups that are on)."""
    return lambda train, y, test: _gradient_boosting(train, y, test, groups)


def gradient_boosting(train, y, test) -> np.ndarray:
    return _gradient_boosting(train, y, test, ())


def _gradient_boosting(train, y, test, groups) -> np.ndarray:
    train, y = _training_set(train, y)
    # columns with fewer than two distinct training values carry no signal (and trip sklearn's binning)
    cols = [c for c in feature_columns(train, groups) if train[c].astype(float).nunique() >= 2]

    _guard_columns(cols)

    def matrix(df):
        return df[cols].to_numpy(dtype=float)  # nulls become NaN, handled natively

    return _symmetric(lambda: HistGradientBoostingClassifier(**HGB_PARAMS), matrix)(train, y, test)


def market(train, y, test) -> np.ndarray:
    """The betting market: the odds-implied probability that fighter 1 wins (margin removed).
    No training. `test` must carry `p_market_f1`; fights without odds are NaN. Not in `MODELS`:
    it is scored only on fights with odds, and no other model ever sees the column."""
    return test["p_market_f1"].to_numpy(dtype=float)


MODELS = {
    "coin_flip": coin_flip,
    "elo": elo_expected,
    "logreg": logistic_regression,
    "hgb": gradient_boosting,
}


def models_for(groups: tuple[str, ...] = ()) -> dict:
    """The models for a feature set. The base set is `MODELS` itself (looked up at call time); a
    larger set swaps in `logreg` and `hgb` with the extra columns. `blend` is not a model here: it is
    the mean of the logits of `hgb` and `logreg`, taken on their predictions (see `blend`)."""
    if not groups:
        return dict(MODELS)
    return {
        **MODELS,
        "logreg": make_logistic_regression(groups),
        "hgb": make_gradient_boosting(groups),
    }


def blend(p_hgb, p_logreg) -> np.ndarray:
    """`blend`: the mean of the logits of `hgb` and logistic regression. Swapping the fighters gives
    1 - p, as for both inputs."""
    z = (_logit(p_hgb) + _logit(p_logreg)) / 2.0
    return 1.0 / (1.0 + np.exp(-z))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-12, 1 - 1e-12)
    return np.log(p / (1 - p))


def model_names(pred: pd.DataFrame) -> list[str]:
    """The models with a `p_<name>` column in a prediction table, in report order (`blend` last)."""
    return [m for m in (*MODELS, "blend") if f"p_{m}" in pred.columns]
