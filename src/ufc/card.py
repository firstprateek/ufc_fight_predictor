"""`ufc card`: the next UFC card with betting prices, written as a card file for `ufc predict --card`.

The source is The Odds API v4 (`GET /v4/sports/mma_mixed_martial_arts/odds`). From it only the two
names, the start time and the bookmaker prices are read (design principle 9). The feed covers every
MMA promotion and does not say which one a fight belongs to, so a fight is kept only when both names
match exactly one fighter in our `fighters` table. Nothing is guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import quote
from zoneinfo import ZoneInfo

import httpx
import pandas as pd

from .odds import OVERROUND_RANGE, remove_margin
from .predict import (
    CARD_OPTIONAL,
    CARD_REQUIRED,
    PredictError,
    _check_date,
    _check_odds,
    _last_fight,
    norm_name,
    resolve_fighter,
)

API_URL = "https://api.the-odds-api.com/v4/sports/mma_mixed_martial_arts/odds"
KEY_ENV = "ODDS_API_KEY"
PARAMS = {"regions": "us", "markets": "h2h", "oddsFormat": "american"}
TIMEOUT = 30
USER_AGENT = "ufc-pipeline/0.1 (research)"
RAW_SUBDIR = "odds_api"
CARD_TZ = ZoneInfo("America/New_York")
MIN_FIGHTS = 5  # the card is the earliest date with at least this many usable fights
DEFAULT_DAYS = 10
NEAR_DAYS = 1  # leftovers are reported for the card date and this many days either side
CARD_COLUMNS = CARD_REQUIRED + CARD_OPTIONAL
NO_KEY_HELP = (
    f"{KEY_ENV} is not set. Get a free key (500 requests a month) by signing up at "
    f"https://the-odds-api.com, then add `export {KEY_ENV}=<your key>` to your shell profile "
    "and open a new shell. (`--from-file` reads a saved response and needs no key.)"
)


class CardError(Exception):
    pass


def parse_now(s: str) -> datetime:
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=UTC)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a UTC time like 2026-11-18T12:00: {s!r}") from None


# ---------------------------------------------------------------- the request


def get_key(env=None) -> str:
    key = (os.environ if env is None else env).get(KEY_ENV, "").strip()
    if not key:
        raise CardError(NO_KEY_HELP)
    return key


def scrub(text: str, key: str) -> str:
    """`text` with the key (raw and URL-encoded) removed."""
    for k in {key, quote(key, safe=""), quote(key)}:
        if k:
            text = text.replace(k, "***")
    return text


def new_client() -> httpx.Client:
    return httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT)


def _api_message(r: httpx.Response) -> str:
    try:
        body = r.json()
    except ValueError:
        body = None
    if isinstance(body, dict) and isinstance(body.get("message"), str):
        return body["message"]
    return r.text.strip()[:200] or "no message"


def fetch_response(key: str, client: httpx.Client) -> tuple[bytes, str | None]:
    """One request. Returns (raw body, x-requests-remaining). Any failure is a `CardError` whose text
    has the key removed; the original exception (whose URL carries the key) is not chained."""
    try:
        r = client.get(API_URL, params={"apiKey": key, **PARAMS})
    except httpx.TimeoutException:
        raise CardError(
            f"the request to The Odds API timed out after {TIMEOUT} s; try again"
        ) from None
    except httpx.HTTPError as e:
        raise CardError(
            "the request to The Odds API failed: " + scrub(f"{type(e).__name__}: {e}", key)
        ) from None
    if r.status_code != 200:
        msg = scrub(_api_message(r), key)
        hint = {
            401: " (check the key in " + KEY_ENV + ")",
            429: " (rate limited or out of credits; wait and retry, see your usage on the site)",
        }.get(r.status_code, "")
        raise CardError(f"The Odds API returned HTTP {r.status_code}: {msg}{hint}")
    return r.content, r.headers.get("x-requests-remaining")


def save_raw(raw_dir: Path, body: bytes, fetched: datetime) -> Path:
    d = raw_dir / RAW_SUBDIR
    d.mkdir(parents=True, exist_ok=True)
    dest = d / f"{fetched.astimezone(UTC):%Y-%m-%dT%H%M%SZ}.json"
    dest.write_bytes(body)
    return dest


def parse_body(body: bytes | str, where: str) -> list:
    try:
        data = json.loads(body)
    except ValueError:
        raise CardError(f"{where} is not JSON") from None
    if not isinstance(data, list):
        raise CardError(f"{where} is not a list of events (got {type(data).__name__})")
    return data


# ---------------------------------------------------------------- reading events


@dataclass
class Event:
    id: str
    start: datetime  # UTC
    name1: str  # home_team
    name2: str  # away_team
    pairs: list[tuple[float, float]]  # per bookmaker that prices both: (price_1, price_2)
    n_bookmakers: int = 0
    max_priced: int = 0  # most priced outcomes any bookmaker lists

    @property
    def label(self) -> str:
        return f"{self.name1} vs {self.name2}"

    @property
    def local_date(self) -> str:
        return self.start.astimezone(CARD_TZ).date().isoformat()


def _parse_time(s) -> datetime:
    if not isinstance(s, str):
        raise ValueError("no commence_time")
    t = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)


def _price(x) -> float | None:
    if isinstance(x, bool) or not isinstance(x, int | float):
        return None
    return float(x) if abs(x) >= 100 else None


def parse_event(e) -> Event:
    """One feed event. Reads only id, commence_time, home_team, away_team and the h2h prices."""
    if not isinstance(e, dict):
        raise ValueError("not an object")
    n1, n2 = e.get("home_team"), e.get("away_team")
    if not (isinstance(n1, str) and isinstance(n2, str) and n1.strip() and n2.strip()):
        raise ValueError("no home_team / away_team")
    ev = Event(
        str(e.get("id", "?")), _parse_time(e.get("commence_time")), n1.strip(), n2.strip(), []
    )
    books = e.get("bookmakers")
    for b in books if isinstance(books, list) else []:
        ev.n_bookmakers += 1
        markets = b.get("markets") if isinstance(b, dict) else None
        for m in markets if isinstance(markets, list) else []:
            if not (isinstance(m, dict) and m.get("key") == "h2h"):
                continue
            priced: dict[str, list[float]] = {}
            for o in m.get("outcomes") if isinstance(m.get("outcomes"), list) else []:
                p = _price(o.get("price")) if isinstance(o, dict) else None
                if p is not None:
                    priced.setdefault(norm_name(o.get("name")), []).append(p)
            ev.max_priced = max(ev.max_priced, len(priced))
            w1, w2 = norm_name(n1), norm_name(n2)
            if w1 != w2 and len(priced.get(w1, [])) == 1 and len(priced.get(w2, [])) == 1:
                ev.pairs.append((priced[w1][0], priced[w2][0]))
    return ev


# ---------------------------------------------------------------- prices


def market_probability(pairs: list[tuple[float, float]]) -> tuple[float | None, int]:
    """Median over bookmakers of the margin-free probability of the first fighter, and how many
    bookmakers it rests on. A bookmaker whose total is outside `OVERROUND_RANGE` is skipped."""
    lo, hi = OVERROUND_RANGE
    ps = []
    for a, b in pairs:
        p1, _, total = remove_margin(a, b)
        if lo <= round(total, 9) <= hi:
            ps.append(p1)
    return (statistics.median(ps) if ps else None), len(ps)


def american_pair(p: float) -> tuple[str, str]:
    """The margin-free American prices that imply probability p for the first fighter. Two decimals
    recover p to well under 0.001."""
    if not 0 < p < 1:
        raise CardError(f"probability {p} is outside (0, 1)")

    def price(q: float) -> float:
        return -100 * q / (1 - q) if q >= 0.5 else 100 * (1 - q) / q

    return f"{price(p):.2f}", f"{price(1 - p):.2f}"


# ---------------------------------------------------------------- choosing the card


@dataclass
class Row:
    id1: str
    id2: str
    name1: str
    name2: str
    date: str
    start: datetime
    p1: float | None
    n_books: int
    odds: tuple[str, str] | None
    note: str | None = None


@dataclass
class Left:
    date: str
    label: str
    reason: str


@dataclass
class Plan:
    card_date: str
    rows: list[Row]
    left: list[Left]  # near the card date
    usable_by_date: dict[str, int]
    left_elsewhere: int = 0
    outside_window: int = 0
    notes: list[str] = field(default_factory=list)


def _classify(ev: Event, tables: dict[str, pd.DataFrame]) -> Row | str:
    """A Row for a usable fight, else the reason it is not."""
    fighters, fights = tables["fighters"], tables["fights"]
    ids, errors = [], []
    for which, name in ((1, ev.name1), (2, ev.name2)):
        try:
            ids.append(resolve_fighter(name, None, which, fighters, fights))
        except PredictError as e:
            errors.append(str(e))
    if errors:
        return "\n".join(errors)
    (id1, n1), (id2, n2) = ids
    if id1 == id2:
        return f"both names are {n1} (the same fighter)"
    if ev.n_bookmakers and ev.max_priced < 2:
        return "fewer than two priced outcomes"
    p, n = market_probability(ev.pairs)
    odds = None
    if p is not None:
        odds = american_pair(p)
        _check_odds([float(x) for x in odds])  # what `ufc predict` will do with them
    return Row(id1, id2, n1, n2, ev.local_date, ev.start, p, n, odds)


def plan_card(
    events: list,
    tables: dict[str, pd.DataFrame],
    now: datetime,
    days: int = DEFAULT_DAYS,
    date: str | None = None,
    min_fights: int = MIN_FIGHTS,
) -> Plan:
    horizon = now + timedelta(days=days)
    rows: dict[str, list[Row]] = {}
    left: list[Left] = []
    seen: set[tuple] = set()
    outside = 0
    for raw in events:
        try:
            ev = parse_event(raw)
        except (ValueError, TypeError) as e:
            eid = raw.get("id", "?") if isinstance(raw, dict) else "?"
            left.append(Left("", f"event {eid}", f"unreadable ({e})"))
            continue
        if not now <= ev.start <= horizon:
            outside += 1
            continue
        res = _classify(ev, tables)
        if isinstance(res, Row):
            key = (res.date, *sorted((res.id1, res.id2)))
            if key in seen:
                left.append(
                    Left(ev.local_date, ev.label, "the same pairing is already on this date")
                )
                continue
            seen.add(key)
            rows.setdefault(res.date, []).append(res)
        else:
            left.append(Left(ev.local_date, ev.label, res))
    dates = sorted({*rows, *(x.date for x in left if x.date)})
    counts = {d: len(rows.get(d, [])) for d in dates}
    if date is not None:
        chosen = date if counts.get(date) else None
    else:
        chosen = next((d for d in dates if counts[d] >= min_fights), None)
    if chosen is None:
        want = f"on {date}" if date else f"on any date with at least {min_fights}"
        shown = "\n".join(f"  {d}: {n} usable" for d, n in counts.items()) or "  (no events)"
        raise CardError(
            f"no card: no usable fights {want} within {days} days of {now:%Y-%m-%d %H:%M} UTC. "
            f"Dates in the feed (New York calendar) and their usable fights:\n{shown}"
        )
    near = {
        (datetime.fromisoformat(chosen) + timedelta(days=k)).date().isoformat()
        for k in range(-NEAR_DAYS, NEAR_DAYS + 1)
    }
    kept = [x for x in left if x.date in near or not x.date]
    out = sorted(rows[chosen], key=lambda r: (r.start, r.name1))
    past = tables["fights"][tables["fights"].date < chosen]
    for r in out:
        cls = []
        for i in (r.id1, r.id2):
            last = _last_fight(past, i)
            cls.append(None if last is None else last.weight_class)
        if cls[0] is None or cls[0] != cls[1]:
            seen_cls = ", ".join(
                f"{n} {c or 'no UFC fight'}" for n, c in zip((r.name1, r.name2), cls, strict=True)
            )
            r.note = f"last fought in different classes ({seen_cls}): `ufc predict` will ask for weight_class"
    return Plan(chosen, out, kept, counts, len(left) - len(kept), outside)


# ---------------------------------------------------------------- the file and the report


def write_card(path: Path, rows: list[Row], force: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("w" if force else "x", newline="", encoding="utf-8")
    except FileExistsError:
        raise CardError(f"{path} already exists; pass --force to overwrite it") from None
    with f:
        w = csv.DictWriter(f, fieldnames=CARD_COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "fighter_1": r.name1,
                    "fighter_2": r.name2,
                    "fighter_1_id": r.id1,
                    "fighter_2_id": r.id2,
                    "date": r.date,
                    "odds_1": r.odds[0] if r.odds else "",
                    "odds_2": r.odds[1] if r.odds else "",
                }
            )


def render(plan: Plan, out: Path, source: str, now: datetime, remaining: str | None) -> str:
    L = [f"Card date: {plan.card_date} (America/New_York)", source]
    if remaining is not None:
        L.append(f"Requests remaining this month: {remaining}")
    L += ["", f"Wrote {out}: {len(plan.rows)} fights"]
    for r in plan.rows:
        if r.p1 is None:
            price = "no usable bookmaker price, prices left blank"
        else:
            price = (
                f"market {r.p1:.1%} for {r.name1}, {r.n_books} bookmaker{'s' * (r.n_books != 1)}"
            )
        L.append(f"  {r.name1} vs {r.name2}: {price}")
        if r.note:
            L.append(f"    note: {r.note}")
    L += ["", f"Left out, on or within {NEAR_DAYS} day of {plan.card_date}: {len(plan.left)}"]
    for x in plan.left:
        body = x.reason.replace("\n", "\n      ")
        L.append(f"  {x.label} ({x.date or 'no date'}): {body}")
    if plan.left_elsewhere:
        L.append(f"  ({plan.left_elsewhere} more left out on other dates, not shown)")
    others = {d: n for d, n in plan.usable_by_date.items() if d != plan.card_date and n}
    if others:
        L.append(
            "Other dates in the window, usable fights: "
            + ", ".join(f"{d} ({n})" for d, n in others.items())
        )
    L += [
        "",
        "Prices are a snapshot at fetch time, not closing prices.",
        "The feed covers every MMA promotion: check that this is the UFC card before logging predictions.",
        "",
        f"Next: uv run ufc predict --card {out} --log   (at least a day before the event)",
    ]
    return "\n".join(L)


def run_card(args, raw: Path, tables: dict, client_factory=None) -> str:
    """Everything `ufc card` does; returns the report. Raises `CardError`."""
    now = args.now or datetime.now(UTC)
    if args.days < 1:
        raise CardError("--days must be at least 1")
    date = _check_date(args.date) if args.date else None
    explicit = args.out is not None
    if explicit and args.out.exists() and not args.force:
        raise CardError(f"{args.out} already exists; pass --force to overwrite it")
    remaining = None
    if args.from_file:
        try:
            body = args.from_file.read_bytes()
        except OSError as e:
            raise CardError(f"cannot read {args.from_file}: {e}") from None
        events = parse_body(body, str(args.from_file))
        source = f"Source: saved response {args.from_file.name} (no request made); window measured from {now:%Y-%m-%d %H:%M} UTC"
    else:
        key = get_key()
        client = (client_factory or new_client)()
        try:
            body, remaining = fetch_response(key, client)
        finally:
            client.close()
        saved = save_raw(raw, body, now)
        events = parse_body(body, "the response")
        source = f"Fetched {now:%Y-%m-%d %H:%M:%S} UTC, raw response kept in {saved}"
    plan = plan_card(events, tables, now, args.days, date)
    out = args.out or args.data_dir.parent / "cards" / f"{plan.card_date}.csv"
    write_card(out, plan.rows, args.force)
    return render(plan, out, source, now, remaining)
