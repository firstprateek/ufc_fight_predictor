"""The evaluation protocol (design principle 8): what may be looked at, and when.

The test period is split in two. Every choice (feature groups kept, anything else) is made on the
development period only. The holdout is run once, after the configuration is frozen in a commit.
"""

from __future__ import annotations

from datetime import date

from ..features import BASE, SET_NAMES, parse_feature_set

TEST_START = "2015-01-01"
DEV_END = "2022-12-31"
HOLDOUT_START = "2023-01-01"

# The frozen configuration: the feature set `ufc backtest` runs by default. None until it is
# frozen. Until then nothing but `base` may be run past DEV_END; after, only `base` and this set.
FINAL_FEATURE_SET: str | None = "all"

# the sets of the development ablation, in table order
ABLATION_SETS = SET_NAMES


class ProtocolError(Exception):
    """A run the protocol does not allow (it would look at the holdout too early)."""


def check_date(name: str, value: str) -> None:
    # dates are compared as strings, so only the canonical form is allowed
    try:
        ok = date.fromisoformat(value).isoformat() == value
    except ValueError:
        ok = False
    if not ok:
        raise ProtocolError(f"{name} must be YYYY-MM-DD, got {value!r}")


_UNSET = object()


def check_run(features_set: str, end: str | None, final=_UNSET) -> None:
    """Refuse a run that would show holdout results for a configuration that is not frozen.

    `base` may always run (its holdout numbers are already published). Anything else may run past
    DEV_END only when it is the frozen configuration. `final` defaults to FINAL_FEATURE_SET."""
    final = FINAL_FEATURE_SET if final is _UNSET else final
    if end is not None:
        check_date("--end", end)
    if parse_feature_set(features_set) == () or (end is not None and end <= DEV_END):
        return
    if final is None:
        raise ProtocolError(
            f"feature set {features_set!r} is not frozen: until a commit freezes the configuration, "
            f"it may only run on the development period (pass --end {DEV_END} or earlier)"
        )
    if parse_feature_set(features_set) != parse_feature_set(final):
        raise ProtocolError(
            f"only the frozen configuration ({final!r}) and {BASE!r} may run past {DEV_END}, "
            f"not {features_set!r}"
        )
