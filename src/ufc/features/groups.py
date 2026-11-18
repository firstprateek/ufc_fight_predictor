"""Feature groups and feature sets, for ablations.

A feature column is `f1_<name>`, `f2_<name>` or `diff_<name>`. The group is decided by `<name>` alone:

- `ratings`: `glicko_*` and `welo` (Glicko-1, method-weighted Elo)
- `adjusted`: `adj_*` (opponent-adjusted offence / defence per dimension)
- `context`: `ctx_*` (strength of schedule, activity and setting)
- `base`: everything else (the features that existed before the groups)

A feature set is `base` plus any of the other groups. Names: `base`, `ratings`, `adjusted`,
`context`, `all`, or a `+` join of groups such as `ratings+adjusted` (`base` is always included).
"""

from __future__ import annotations

BASE = "base"
GROUPS = ("ratings", "adjusted", "context")
SET_NAMES = (BASE, *GROUPS, "all")

# the five dimensions of the opponent-adjusted model
DIMENSIONS = ("slpm", "td15", "ctrl_share", "kd15", "sub15")

# extra logistic-regression inputs per group (added to its base inputs when the group is on)
LOGREG_EXTRA = {
    "ratings": ("diff_glicko_r", "diff_welo"),
    "adjusted": tuple(f"diff_adj_{d}_exp" for d in DIMENSIONS),
    "context": (),
}


class UnknownFeatureSet(ValueError):
    pass


def column_group(col: str) -> str:
    """The group of a feature column (`base` for any column that is not in a named group)."""
    name = col.split("_", 1)[1] if col.startswith(("f1_", "f2_", "diff_")) else col
    if name.startswith("glicko_") or name == "welo":
        return "ratings"
    if name.startswith("adj_"):
        return "adjusted"
    if name.startswith("ctx_"):
        return "context"
    return BASE


def parse_feature_set(spec: str) -> tuple[str, ...]:
    """`all` -> every group; `ratings+adjusted` -> those two. Returns the groups on (base is implied),
    in canonical order. An empty tuple means `base` alone."""
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    if not parts:
        raise UnknownFeatureSet(f"empty feature set {spec!r}")
    on: set[str] = set()
    for p in parts:
        if p == BASE:
            continue
        if p == "all":
            on.update(GROUPS)
        elif p in GROUPS:
            on.add(p)
        else:
            raise UnknownFeatureSet(f"unknown feature set {p!r}; use one of {', '.join(SET_NAMES)}")
    return tuple(g for g in GROUPS if g in on)


def set_name(groups: tuple[str, ...]) -> str:
    """Canonical name of a feature set: `base`, one group, `all`, or a `+` join."""
    if not groups:
        return BASE
    if set(groups) == set(GROUPS):
        return "all"
    return "+".join(groups)


def in_set(col: str, groups: tuple[str, ...]) -> bool:
    return column_group(col) in (BASE, *groups)


def check_columns(features, groups: tuple[str, ...]) -> None:
    """A feature set may only run when its columns exist (an old features.parquet has none)."""
    for g in groups:
        have = [c for c in features.columns if column_group(c) == g]
        need = [c for c in LOGREG_EXTRA[g] if c not in features.columns]
        if not have or need:
            raise UnknownFeatureSet(
                f"features have no usable `{g}` columns (missing {need or 'all'}): "
                "rebuild with `ufc features`"
            )
