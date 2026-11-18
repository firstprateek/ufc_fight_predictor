"""`ufc card`, offline. The response fixture (tests/fixtures/odds_api/response.json) is written by hand
in the documented shape of The Odds API v4 `GET /v4/sports/{sport}/odds`: real fighter names, and
prices that are MADE UP. It is not fetched data and contains no real price."""

import csv
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest
from test_predict import lightweights, synth_snapshot, world  # noqa: F401

from ufc import card as C
from ufc import predict as P
from ufc.cli import main
from ufc.odds import implied_prob, remove_margin

FIXTURE = Path(__file__).parent / "fixtures" / "odds_api" / "response.json"
NOW = datetime(2026, 11, 18, 12, 0, tzinfo=UTC)
NOW_ARG = "2026-11-18T12:00"
KEY = "sekrit-key-0123456789abcdef"
NOT_IN_TABLE = {
    "Dustin Poirer",
    "Madeup Bantam",
    "Another Madeup",
    "Madeup Heavy",
    "Other Madeup",
    "Bruno Silva",
}


def events():
    return json.loads(FIXTURE.read_text())


def tables():
    """Fighters table with every fixture name that should match, two Bruno Silvas, one past fight each."""
    names = {n for e in events() for n in (e["home_team"], e["away_team"])}
    names = sorted(
        n for n in names | {"Dustin Poirier", "Jack Hermansson"} if n not in NOT_IN_TABLE
    )
    names = [n for n in names if n != "zhang  weili"] + ["Bruno Silva", "Bruno Silva"]
    fighters = pd.DataFrame(
        {"fighter_id": [f"id{i:02d}" for i in range(len(names))], "name": names}
    )
    rows = [
        {
            "fight_id": f"p{i}",
            "date": "2026-01-01",
            "bout_order": 1,
            "fighter_1_id": f.fighter_id,
            "fighter_2_id": f"dummy{i}",
            "winner_id": f.fighter_id,
            "outcome": "win",
            "weight_class": "Catch Weight" if f["name"] == "Cesar Almeida" else "Lightweight",
        }
        for i, f in fighters.iterrows()
    ]
    return {"fighters": fighters, "fights": pd.DataFrame(rows)}


def plan(**kw):
    return C.plan_card(events(), tables(), NOW, **kw)


def pair(p):
    return tuple(float(x) for x in C.american_pair(p))


# ---------------------------------------------------------------- the request


def mock_client(handler, seen=None):
    def h(request):
        if seen is not None:
            seen.append(request)
        return handler(request)

    return lambda: httpx.Client(transport=httpx.MockTransport(h), timeout=C.TIMEOUT)


def ok(request):
    return httpx.Response(
        200, content=FIXTURE.read_bytes(), headers={"x-requests-remaining": "412"}
    )


def run_cli(tmp_path, *extra):
    return main(["--data-dir", str(tmp_path / "data"), "card", *extra])


@pytest.fixture
def env(tmp_path, monkeypatch):
    t = tables()
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    monkeypatch.setattr("ufc.cli.load_tables", lambda _p: t)
    monkeypatch.setenv(C.KEY_ENV, KEY)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_request_path_parameters_and_key_placement(env, monkeypatch, capsys):
    seen = []
    monkeypatch.setattr(C, "new_client", mock_client(ok, seen))
    assert run_cli(env, "--now", NOW_ARG) == 0
    (r,) = seen
    assert r.method == "GET"
    assert r.url.host == "api.the-odds-api.com"
    assert r.url.path == "/v4/sports/mma_mixed_martial_arts/odds"
    assert dict(r.url.params) == {
        "apiKey": KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american",
    }
    assert KEY not in str(r.headers) and KEY not in r.url.path
    out = capsys.readouterr().out
    assert "Requests remaining this month: 412" in out and KEY not in out
    saved = list((env / "data" / "raw" / "odds_api").glob("*.json"))
    assert len(saved) == 1 and re.fullmatch(r"\d{4}-\d\d-\d\dT\d{6}Z\.json", saved[0].name)
    assert saved[0].read_bytes() == FIXTURE.read_bytes()
    assert (env / "cards" / "2026-11-21.csv").exists()


def test_the_key_is_written_nowhere(env, monkeypatch):
    monkeypatch.setattr(C, "new_client", mock_client(ok))
    assert run_cli(env, "--now", NOW_ARG) == 0
    for p in env.rglob("*"):
        if p.is_file():
            assert KEY.encode() not in p.read_bytes(), p


def test_missing_key_explains_and_fails(env, monkeypatch, capsys):
    monkeypatch.delenv(C.KEY_ENV)
    monkeypatch.setattr(C, "new_client", lambda: pytest.fail("no request without a key"))
    assert run_cli(env) == 1
    err = capsys.readouterr().err
    assert "the-odds-api.com" in err and f"export {C.KEY_ENV}=" in err
    monkeypatch.setenv(C.KEY_ENV, "   ")
    assert run_cli(env) == 1


@pytest.mark.parametrize(
    "status,body",
    [
        (401, {"message": f"Invalid key {KEY}", "error_code": "INVALID_KEY"}),
        (429, {"message": "Too many requests", "error_code": "EXCEEDED_FREQ_LIMIT"}),
        (500, "plain text echoing " + KEY),
    ],
)
def test_http_errors_report_status_and_message_without_the_key(
    env, monkeypatch, capsys, status, body
):
    def handler(request):
        assert KEY in str(request.url)  # the url does carry it: it must never be printed
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body)

    monkeypatch.setattr(C, "new_client", mock_client(handler))
    assert run_cli(env) == 1
    cap = capsys.readouterr()
    text = cap.out + cap.err
    assert f"HTTP {status}" in cap.err and KEY not in text
    assert not (env / "cards").exists() and not (env / "data" / "raw" / "odds_api").exists()


def test_a_timeout_and_a_connection_error_do_not_leak_the_key(env, monkeypatch, capsys):
    def timeout(request):
        raise httpx.ReadTimeout(f"timed out {request.url}", request=request)

    def refused(request):
        raise httpx.ConnectError(f"cannot connect to {request.url}", request=request)

    for handler, words in ((timeout, "timed out"), (refused, "ConnectError")):
        monkeypatch.setattr(C, "new_client", mock_client(handler))
        assert run_cli(env) == 1
        cap = capsys.readouterr()
        assert words in cap.err and KEY not in cap.out + cap.err and "Traceback" not in cap.err


def test_scrub_removes_the_raw_and_the_encoded_key():
    key = "a b&c"
    assert key not in C.scrub(f"x {key} y a%20b%26c a+b%26c", key)
    assert "%20" not in C.scrub("a%20b%26c", key)


def test_a_body_that_is_not_a_list_of_events_fails_clearly(env, monkeypatch, capsys):
    monkeypatch.setattr(
        C, "new_client", mock_client(lambda r: httpx.Response(200, json={"message": "x"}))
    )
    assert run_cli(env) == 1
    assert "not a list of events" in capsys.readouterr().err


def test_from_file_needs_no_key_and_makes_no_request(env, monkeypatch, capsys):
    monkeypatch.delenv(C.KEY_ENV)
    monkeypatch.setattr(C, "new_client", lambda: pytest.fail("no request with --from-file"))
    assert run_cli(env, "--from-file", str(FIXTURE), "--now", NOW_ARG) == 0
    assert "no request made" in capsys.readouterr().out
    assert not (env / "data" / "raw" / "odds_api").exists()


# ---------------------------------------------------------------- which fights


def test_fixture_is_hand_made_and_marked():
    ev = events()
    assert all("made up" in e["sport_title"] for e in ev)
    assert all(b["title"].startswith("Made Up ") for e in ev for b in e["bookmakers"])


def test_default_card_is_the_earliest_date_with_five_usable_fights():
    p = plan()
    assert p.card_date == "2026-11-21"  # 11-19 has only 3 usable fights
    assert p.usable_by_date == {"2026-11-19": 3, "2026-11-21": 6, "2026-11-27": 5, "2026-11-22": 0}
    assert C.MIN_FIGHTS == 5


def test_a_card_spanning_midnight_utc_is_one_new_york_date():
    p = plan()
    starts = {r.name1: r.start for r in p.rows}
    assert starts["Casey O'Neill"].date().isoformat() == "2026-11-22"  # 02:30 UTC
    assert starts["Gabriel Bonfim"].date().isoformat() == "2026-11-21"
    assert {r.date for r in p.rows} == {"2026-11-21"}
    # 05:00 UTC is midnight in New York (EST): the next calendar date
    assert C.parse_event(next(e for e in events() if e["id"] == "e21")).local_date == "2026-11-22"


def test_date_chooses_a_card_even_with_fewer_than_five():
    p = plan(date="2026-11-19")
    assert p.card_date == "2026-11-19" and len(p.rows) == 3
    assert plan(date="2026-11-27").card_date == "2026-11-27"


def test_days_limits_the_window():
    assert plan(days=30).usable_by_date["2026-12-05"] == 1
    assert "2026-11-27" not in plan(days=9).usable_by_date  # 11-27 22:00Z is past 11-27 12:00Z
    with pytest.raises(C.CardError, match="no usable fights on 2026-12-05 within 10 days"):
        plan(date="2026-12-05")
    # an event that has started is not on the card
    assert all(r.name1 != "Ty Miller" for r in plan(days=30).rows)


def test_no_qualifying_date_lists_dates_and_writes_nothing(env, capsys):
    with pytest.raises(C.CardError) as e:
        plan(days=2)
    assert "2026-11-19: 3 usable" in str(e.value)
    assert run_cli(env, "--from-file", str(FIXTURE), "--now", NOW_ARG, "--days", "2") == 1
    err = capsys.readouterr().err
    assert "no card" in err and "2026-11-19: 3 usable" in err
    assert not (env / "cards").exists()
    assert run_cli(env, "--from-file", str(FIXTURE), "--now", NOW_ARG, "--date", "2026-12-25") == 1
    assert not (env / "cards").exists()


def left(p, label):
    (x,) = [x for x in p.left if x.label == label]
    return x.reason


def test_what_is_left_out_and_why():
    p = plan()
    assert {r.name1 for r in p.rows} == {
        "Gabriel Bonfim",
        "Arnold Allen",
        "Chris Padilla",
        "Jiri Prochazka",
        "Casey O'Neill",
        "Cesar Almeida",
    }
    r = left(p, "Dustin Poirer vs Mark Choinski")
    assert "no fighter called 'Dustin Poirer'" in r and "Dustin Poirier" in r  # closest names
    r = left(p, "Bruno Silva vs Jack Hermansson")
    assert "2 fighters are called 'Bruno Silva'" in r and r.count("UFC record") == 2
    assert "id38" in r and "id39" in r  # the candidates come with their ids
    assert "same fighter" in left(p, "Zhang Weili vs zhang  weili")
    assert left(p, "Irina Alekseeva vs Daria Zhelezniakova") == "fewer than two priced outcomes"
    assert "Madeup Bantam" in left(p, "Madeup Bantam vs Another Madeup")
    assert "Another Madeup" in left(p, "Madeup Bantam vs Another Madeup")  # both names reported
    assert p.left_elsewhere == 0


def test_the_same_pairing_twice_on_a_date_is_written_once():
    ev = events() + [
        dict(
            next(e for e in events() if e["id"] == "e10"),
            id="dup",
            home_team="Michael Oliveira",
            away_team="Gabriel Bonfim",
        )
    ]
    p = C.plan_card(ev, tables(), NOW)
    assert sum(r.name1 in ("Gabriel Bonfim", "Michael Oliveira") for r in p.rows) == 1
    assert "already on this date" in left(p, "Michael Oliveira vs Gabriel Bonfim")


def test_an_unreadable_event_is_reported_not_fatal():
    p = C.plan_card(events() + [{"id": "bad"}, "junk"], tables(), NOW)
    assert any("unreadable" in x.reason for x in p.left) and len(p.rows) == 6


def test_names_are_matched_by_the_project_rules_and_written_as_in_the_table():
    e = dict(next(e for e in events() if e["id"] == "e10"), home_team="  GABRIEL  BONFIM ")
    p = C.plan_card([e], tables(), NOW, date="2026-11-21")
    assert p.rows[0].name1 == "Gabriel Bonfim"


def test_a_fight_whose_fighters_last_fought_in_different_classes_is_noted():
    notes = {r.name1: r.note for r in plan().rows}
    assert "ask for weight_class" in notes["Cesar Almeida"]
    assert notes["Gabriel Bonfim"] is None


# ---------------------------------------------------------------- prices


def row(name1):
    return {r.name1: r for r in plan().rows}[name1]


def test_median_over_an_odd_number_of_bookmakers():
    # Bonfim: -300/+250 -> .75/(.75+.285714)=.724138; -110/-110 -> .5; -200/+150 -> .666667/1.066667=.625
    r = row("Gabriel Bonfim")
    assert r.n_books == 3 and r.p1 == pytest.approx(
        0.625, abs=1e-9
    )  # the middle one, not the mean (.616)


def test_median_over_an_even_number_of_bookmakers():
    # Allen: .5, .625, .724138, .5 -> sorted .5 .5 .625 .724138 -> (.5+.625)/2
    r = row("Arnold Allen")
    assert r.n_books == 4 and r.p1 == pytest.approx(0.5625, abs=1e-9)


def test_a_bookmaker_with_the_outcomes_in_the_other_order_is_matched_by_name():
    # bk_a: Almeida -150, Cepo +130. bk_b: Almeida +130, Cepo -150, listed Cepo first. Read by
    # position the second would give .5798 as well; by name it gives .4202, so the median is .5
    r = row("Cesar Almeida")
    a, b = remove_margin(-150, 130)[0], remove_margin(130, -150)[0]
    assert r.n_books == 2 and r.p1 == pytest.approx((a + b) / 2, abs=1e-9)


def test_bookmakers_with_a_bad_total_are_skipped():
    # -300/-300 totals 1.5 and +120/+120 totals .909, both outside the range; -110/-110 stays
    r = row("Chris Padilla")
    assert r.n_books == 1 and r.p1 == pytest.approx(0.5, abs=1e-12)


def test_a_fight_with_no_bookmaker_is_written_with_blank_prices(tmp_path):
    r = row("Jiri Prochazka")
    assert r.p1 is None and r.odds is None
    out = tmp_path / "c.csv"
    C.write_card(out, plan().rows)
    got = {x["fighter_1"]: x for x in csv.DictReader(out.open())}
    assert got["Jiri Prochazka"]["odds_1"] == "" and got["Jiri Prochazka"]["odds_2"] == ""
    assert P.load_card(out)[3].odds is None


@pytest.mark.parametrize(
    "p", [0.5, 0.5625, 0.724138, 0.3, 0.1234567, 0.01, 0.99, 0.4999999, 0.5000001]
)
def test_written_prices_pass_check_odds_and_round_trip(p):
    a, b = C.american_pair(p)
    assert P._check_odds([float(a), float(b)]) is not None
    assert remove_margin(float(a), float(b))[0] == pytest.approx(p, abs=1e-3)
    assert implied_prob(float(a)) + implied_prob(float(b)) == pytest.approx(1.0, abs=1e-3)


def test_written_prices_of_the_whole_fixture_round_trip():
    for r in plan().rows:
        if r.odds:
            assert remove_margin(*(float(x) for x in r.odds))[0] == pytest.approx(r.p1, abs=1e-3)


# ---------------------------------------------------------------- the file


def test_file_has_exactly_the_card_columns_and_loads(tmp_path):
    out = tmp_path / "c.csv"
    p = plan()
    C.write_card(out, p.rows)
    rows = list(csv.DictReader(out.open()))
    assert list(rows[0]) == [
        "fighter_1",
        "fighter_2",
        "fighter_1_id",
        "fighter_2_id",
        "date",
        "weight_class",
        "scheduled_rounds",
        "is_title",
        "odds_1",
        "odds_2",
    ]
    assert all(r["weight_class"] == r["scheduled_rounds"] == r["is_title"] == "" for r in rows)
    reqs = P.load_card(out)
    assert len(reqs) == 6 and all(r.date == "2026-11-21" and r.id1 and r.id2 for r in reqs)
    assert reqs[0].odds is not None


def test_no_overwrite_without_force(env, capsys):
    args = ("--from-file", str(FIXTURE), "--now", NOW_ARG)
    assert run_cli(env, *args) == 0
    path = env / "cards" / "2026-11-21.csv"
    first = path.read_text()
    assert run_cli(env, *args) == 1
    assert "--force" in capsys.readouterr().err and path.read_text() == first
    path.write_text("changed")
    assert run_cli(env, *args, "--force") == 0 and path.read_text() == first
    out = env / "elsewhere" / "x.csv"
    assert run_cli(env, *args, "--out", str(out)) == 0
    assert run_cli(env, *args, "--out", str(out)) == 1  # checked before any request is made


def test_the_report_has_the_required_content(env, capsys):
    assert run_cli(env, "--from-file", str(FIXTURE), "--now", NOW_ARG) == 0
    out = capsys.readouterr().out
    assert "Card date: 2026-11-21" in out
    assert "market 62.5% for Gabriel Bonfim, 3 bookmakers" in out
    assert "no usable bookmaker price" in out
    assert "not closing prices" in out and "every MMA promotion" in out
    assert "Next: uv run ufc predict --card" in out and "--log" in out
    assert "(1 more" not in out


# ---------------------------------------------------------------- through predict


def test_the_written_file_runs_through_predict(world, tmp_path, monkeypatch, capsys):  # noqa: F811
    tabs, day = world
    synth_snapshot(tabs, tmp_path / "data" / "processed")
    monkeypatch.chdir(tmp_path)
    a, b, c, d = lightweights(*world)[:4]
    n = tabs["fighters"].set_index("fighter_id").name
    start = datetime.fromisoformat(day).replace(hour=20, tzinfo=UTC)

    def ev(i, x, y, prices):
        bms = [
            {
                "key": f"b{j}",
                "title": "Made Up",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [{"name": n[x], "price": p1}, {"name": n[y], "price": p2}],
                    }
                ],
            }
            for j, (p1, p2) in enumerate(prices)
        ]
        return {
            "id": i,
            "commence_time": start.isoformat().replace("+00:00", "Z"),
            "home_team": n[x],
            "away_team": n[y],
            "bookmakers": bms,
        }

    feed = tmp_path / "feed.json"
    feed.write_text(json.dumps([ev("1", a, b, [(-150, 130), (-140, 120)]), ev("2", c, d, [])]))
    now = (start - timedelta(days=2)).strftime("%Y-%m-%dT%H:%M")
    base = ["--data-dir", str(tmp_path / "data")]
    assert main([*base, "card", "--from-file", str(feed), "--now", now, "--date", day]) == 0
    capsys.readouterr()
    card = tmp_path / "cards" / f"{day}.csv"
    assert main([*base, "predict", "--card", str(card), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert len(doc["fights"]) == 2
    m = doc["fights"][0]["market"]
    p_expected = sorted(remove_margin(*x)[0] for x in [(-150, 130), (-140, 120)])
    assert m["p_market_1"] == pytest.approx(sum(p_expected) / 2, abs=1e-3)
    assert "market" not in doc["fights"][1] or doc["fights"][1]["market"] is None


def test_now_with_an_offset_is_converted_and_a_live_fetch_names_the_file_by_the_real_time(
    env, monkeypatch, capsys
):
    assert C.parse_now("2026-11-18T07:00-05:00") == NOW
    monkeypatch.setattr(C, "new_client", mock_client(ok))
    assert run_cli(env, "--now", NOW_ARG) == 0
    (saved,) = (env / "data" / "raw" / "odds_api").glob("*.json")
    assert saved.name != "2026-11-18T120000Z.json"  # the real clock, not --now


def test_an_existing_default_file_after_a_fetch_points_at_the_saved_response(
    env, monkeypatch, capsys
):
    monkeypatch.setattr(C, "new_client", mock_client(ok))
    assert run_cli(env, "--now", NOW_ARG) == 0
    capsys.readouterr()
    assert run_cli(env, "--now", NOW_ARG) == 1
    err = capsys.readouterr().err
    assert "--force" in err and "--from-file" in err


def test_a_card_dated_today_warns_about_logging_late(env, capsys):
    now = "2026-11-21T15:00"  # the fights are later that day, New York time
    assert run_cli(env, "--from-file", str(FIXTURE), "--now", now, "--date", "2026-11-21") == 0
    assert "logged too late" in capsys.readouterr().out
    assert run_cli(env, "--from-file", str(FIXTURE), "--now", NOW_ARG, "--force") == 0
    assert "logged too late" not in capsys.readouterr().out
