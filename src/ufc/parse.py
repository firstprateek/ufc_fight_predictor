"""Raw source cells -> typed values. Pure functions, no I/O."""

from __future__ import annotations

import re
from datetime import datetime

NULL_CELLS = {"", "--", "---"}

_DIVISION_LIMITS_LBS = {
    "Strawweight": 115,
    "Flyweight": 125,
    "Bantamweight": 135,
    "Featherweight": 145,
    "Lightweight": 155,
    "Welterweight": 170,
    "Middleweight": 185,
    "Light Heavyweight": 205,
    "Heavyweight": 265,
}
_DIVISION_RE = re.compile(
    "|".join(sorted(map(re.escape, _DIVISION_LIMITS_LBS), key=len, reverse=True))
)


def clean(cell: str | None) -> str | None:
    """Strip a cell; the source's `--`, `---` and blanks become None."""
    if cell is None:
        return None
    cell = cell.strip()
    return None if cell in NULL_CELLS else cell


def to_int(cell: str | None) -> int | None:
    cell = clean(cell)
    if cell is None:
        return None
    value = float(cell)  # some source cells are written '1.0'
    if value != int(value):
        raise ValueError(f"not an integer: {cell!r}")
    return int(value)


def parse_of(cell: str | None) -> tuple[int | None, int | None]:
    """'41 of 65' -> (41, 65); null cells -> (None, None)."""
    cell = clean(cell)
    if cell is None:
        return None, None
    m = re.fullmatch(r"(\d+) of (\d+)", cell)
    if not m:
        raise ValueError(f"bad 'X of Y' cell: {cell!r}")
    return int(m[1]), int(m[2])


def parse_clock(cell: str | None) -> int | None:
    """'m:ss' -> seconds; '--' -> None."""
    cell = clean(cell)
    if cell is None:
        return None
    m = re.fullmatch(r"(\d+):(\d{2})", cell)
    if not m:
        raise ValueError(f"bad m:ss cell: {cell!r}")
    return int(m[1]) * 60 + int(m[2])


def parse_height_cm(cell: str | None) -> float | None:
    """`5' 11"` -> 180.3."""
    cell = clean(cell)
    if cell is None:
        return None
    m = re.fullmatch(r"(\d+)' ?(\d+)\"", cell)
    if not m:
        raise ValueError(f"bad height: {cell!r}")
    return round((int(m[1]) * 12 + int(m[2])) * 2.54, 1)


def parse_reach_cm(cell: str | None) -> float | None:
    """`70"` -> 177.8."""
    cell = clean(cell)
    if cell is None:
        return None
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\"", cell)
    if not m:
        raise ValueError(f"bad reach: {cell!r}")
    return round(float(m[1]) * 2.54, 1)


def parse_weight_lbs(cell: str | None) -> int | None:
    cell = clean(cell)
    if cell is None:
        return None
    m = re.fullmatch(r"(\d+) lbs\.", cell)
    if not m:
        raise ValueError(f"bad weight: {cell!r}")
    return int(m[1])


def parse_date(cell: str | None) -> str | None:
    """'November 14, 2026' or 'Jul 13, 1978' -> ISO date."""
    cell = clean(cell)
    if cell is None:
        return None
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(cell, fmt).date().isoformat()
        except ValueError:
            continue
    raise ValueError(f"bad date: {cell!r}")


def parse_outcome(cell: str) -> str:
    """Source outcome (in the bout's listed order) -> 'first_won' | 'second_won' | 'draw' | 'nc'."""
    return {"W/L": "first_won", "L/W": "second_won", "D/D": "draw", "NC/NC": "nc"}[cell.strip()]


def parse_scheduled_rounds(cell: str | None) -> int | None:
    """'3 Rnd (5-5-5)' -> 3; '1 Rnd + OT (12-3)' -> 1; 'No Time Limit' -> None."""
    cell = clean(cell)
    m = re.match(r"(\d+) Rnd", cell or "")
    return int(m[1]) if m else None


def parse_round(cell: str | None) -> int | None:
    """'Round 3' or '3' -> 3."""
    cell = clean(cell)
    if cell is None:
        return None
    m = re.fullmatch(r"(?:Round )?(\d+)", cell)
    if not m:
        raise ValueError(f"bad round: {cell!r}")
    return int(m[1])


def parse_weight_class(cell: str) -> dict:
    """'UFC Women's Bantamweight Title Bout' -> division, is_title, gender."""
    raw = cell.strip()
    is_title = "Title" in raw
    gender = "female" if "Women's" in raw else "male"
    m = _DIVISION_RE.search(raw)
    if m:
        division = ("Women's " if gender == "female" else "") + m[0]
    elif "Open Weight" in raw:
        division = "Open Weight"
    elif "Catch Weight" in raw:
        division = "Catch Weight"
    else:
        division = re.sub(r"\s*(Title\s*)?Bout$", "", raw).strip() or raw
    return {"weight_class": division, "is_title": is_title, "gender": gender}


def division_limit_lbs(weight_class: str) -> int | None:
    m = _DIVISION_RE.search(weight_class)
    return _DIVISION_LIMITS_LBS[m[0]] if m else None


def fighter_id(url: str) -> str:
    """Hex id at the end of a ufcstats URL."""
    return url.strip().rstrip("/").rsplit("/", 1)[-1]
