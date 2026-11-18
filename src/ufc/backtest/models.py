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

from ..features import feature_columns, swap_orientation

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


def _logreg_matrix(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    d["diff_sig_diff"] = d["diff_slpm_career"].astype(float) - d["diff_sapm_career"].astype(float)
    return d[LOGREG_DIFFS].to_numpy(dtype=float)


def _symmetric(make, matrix):
    """Wrap a classifier factory in the two-orientation train / averaged predict protocol."""

    def run(train, y, test) -> np.ndarray:
        both = pd.concat([train, swap_orientation(train)], ignore_index=True)
        labels = np.concatenate([y, 1 - y])
        model = make().fit(matrix(both), labels)

        def raw(df):
            return model.predict_proba(matrix(df))[:, 1]

        return (raw(test) + 1.0 - raw(swap_orientation(test))) / 2.0

    return run


def logistic_regression(train, y, test) -> np.ndarray:
    def make():
        return make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),  # train medians + missing flags
            StandardScaler(),
            LogisticRegression(C=LOGREG_C, max_iter=1000),
        )

    return _symmetric(make, _logreg_matrix)(train, y, test)


def gradient_boosting(train, y, test) -> np.ndarray:
    # columns with fewer than two distinct training values carry no signal (and trip sklearn's binning)
    cols = [c for c in feature_columns(train) if train[c].astype(float).nunique() >= 2]

    def matrix(df):
        return df[cols].to_numpy(dtype=float)  # nulls become NaN, handled natively

    return _symmetric(lambda: HistGradientBoostingClassifier(**HGB_PARAMS), matrix)(train, y, test)


MODELS = {
    "coin_flip": coin_flip,
    "elo": elo_expected,
    "logreg": logistic_regression,
    "hgb": gradient_boosting,
}
