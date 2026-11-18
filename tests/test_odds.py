import shutil
from pathlib import Path

import pandas as pd
import pytest

import ufc.odds as O
from ufc.cli import main
from ufc.odds import OddsError

FIX = Path(__file__).parent / "fixtures"
SHA = "b" * 40


def tables(rows):
    """rows: (fight_id, date, name1, name2, outcome). Fighter ids sort like the names."""
    ids = {}
    for _, _, a, b, _ in rows:
        for n in (a, b):
            ids.setdefault(n, f"id{len(ids):03d}")
    fighters = pd.DataFrame({"fighter_id": list(ids.values()), "name": list(ids)})
    recs = []
    for fid, d, a, b, outcome in rows:
        recs.append(
            {"fight_id": fid, "date": d, "fighter_1_id": ids[a], "fighter_2_id": ids[b],
             "outcome": outcome}
        )  # fmt: skip
    return {"fights": pd.DataFrame(recs), "fighters": fighters}


def src(*rows):
    return pd.DataFrame(rows, columns=["R_fighter", "B_fighter", "R_odds", "B_odds", "date"])


def run(t, source, aliases=None):
    cleaned, _ = O.clean_source(source)
    rows, rep, _ = O.match_odds(cleaned, t["fights"], t["fighters"], aliases or {})
    return rows, rep


# --- the five-column rule -------------------------------------------------------------------


def test_loader_reads_only_the_five_columns():
    df = O.load_odds_source(FIX / "odds" / "ufc-master.csv")
    assert list(df.columns) == O.ODDS_COLUMNS
    assert len(df) == 4


def test_loader_asks_pandas_for_exactly_the_five_columns(monkeypatch):
    seen = {}
    real = pd.read_csv

    def spy(*a, **kw):
        seen.update(kw)
        return real(*a, **kw)

    monkeypatch.setattr(O.pd, "read_csv", spy)
    O.load_odds_source(FIX / "odds" / "ufc-master.csv")
    assert sorted(seen["usecols"]) == sorted(O.ODDS_COLUMNS)


def test_loader_never_touches_result_columns(tmp_path):
    # a result column holding unparseable garbage and a duplicate-looking name must not matter
    p = tmp_path / "ufc-master.csv"
    p.write_text(
        "R_fighter,B_fighter,R_odds,B_odds,date,Winner,R_odds_result\n"
        'A B,C D,-110,100,2020-01-01,"x,""y",zzz\n'
    )
    df = O.load_odds_source(p)
    assert list(df.columns) == O.ODDS_COLUMNS and df.R_odds.iloc[0] == -110


def test_loader_requires_the_five_columns(tmp_path):
    p = tmp_path / "ufc-master.csv"
    p.write_text("R_fighter,B_fighter,R_odds,date\nA,B,-110,2020-01-01\n")
    with pytest.raises(OddsError, match="B_odds"):
        O.load_odds_source(p)


# --- prices -----------------------------------------------------------------------------------


def test_implied_prob_and_margin_removal():
    assert O.implied_prob(100) == 0.5
    assert O.implied_prob(-200) == pytest.approx(2 / 3)
    assert O.implied_prob(300) == pytest.approx(0.25)
    t = tables([("f1", "2020-01-01", "A A", "B B", "win")])
    rows, rep = run(t, src(("A A", "B B", -200.0, 170.0, "2020-01-01")))
    df = O.make_odds_table(rows, rep)
    q1, q2 = 2 / 3, 100 / 270
    assert df.p_market_f1.iloc[0] == pytest.approx(q1 / (q1 + q2))
    assert df.overround.iloc[0] == pytest.approx(q1 + q2)


def test_clean_source_drops_blank_and_invalid_prices():
    s = src(("A", "B", None, 100.0, "2020-01-01"), ("A", "B", -50.0, 100.0, "2020-01-01"),
            ("A", "B", -110.0, 100.0, "2020-01-01"))  # fmt: skip
    kept, dropped = O.clean_source(s)
    assert len(kept) == 1 and dropped == 2


# --- matching ---------------------------------------------------------------------------------


def test_normalised_names_match_and_orientation_follows_ours():
    t = tables([("f1", "2020-01-01", "Jose Aldo", "Zed Zane", "win")])
    # odds file: accents, case, suffix, punctuation, spacing; red corner is our fighter_2
    rows, rep = run(t, src(("  ZED  zane ", "José Aldo Jr.", -300.0, 250.0, "2020-01-01")))
    assert rep.matched_by_fallback == 0
    assert rows == [{"fight_id": "f1", "odds_f1": 250.0, "odds_f2": -300.0}]


def test_hyphen_and_apostrophe():
    assert O.normalise_name("Roldan Sangcha'an") == O.normalise_name("Roldan Sangchaan")
    assert O.normalise_name("Abdul-Malik") == O.normalise_name("abdul malik")


def test_date_window_is_one_day():
    t = tables([("f1", "2020-01-10", "A A", "B B", "win")])
    assert len(run(t, src(("A A", "B B", -110.0, 100.0, "2020-01-11")))[0]) == 1
    assert len(run(t, src(("A A", "B B", -110.0, 100.0, "2020-01-09")))[0]) == 1
    rows, rep = run(t, src(("A A", "B B", -110.0, 100.0, "2020-01-12")))
    assert rows == [] and len(rep.unmatched) == 1


def test_fallback_needs_same_day_equal_surnames_and_one_exact_name():
    t = tables([("f1", "2020-01-10", "Nicholas Galanti", "Zed Zane", "win")])
    rows, rep = run(t, src(("Nick Galanti", "Zed Zane", -110.0, 100.0, "2020-01-10")))
    assert len(rows) == 1 and rep.matched_by_fallback == 1
    # a day off: not allowed in the fallback
    assert run(t, src(("Nick Galanti", "Zed Zane", -110.0, 100.0, "2020-01-11")))[0] == []
    # neither name exact: never guess
    assert run(t, src(("Nick Galanti", "Zid Zane", -110.0, 100.0, "2020-01-10")))[0] == []
    # different surname: never guess
    assert run(t, src(("Nick Gallanti", "Zed Zane", -110.0, 100.0, "2020-01-10")))[0] == []


def test_alias_resolves_a_renamed_fighter():
    t = tables([("f1", "2020-01-10", "Katlyn Cerminara", "Zed Zane", "win")])
    s = src(("Katlyn Chookagian", "Zed Zane", -110.0, 100.0, "2020-01-10"))
    assert run(t, s)[0] == []
    assert len(run(t, s, {"katlyn chookagian": "katlyn cerminara"})[0]) == 1


def test_a_fight_claimed_twice_stays_unmatched():
    t = tables([("f1", "2020-01-10", "A A", "B B", "win")])
    s = src(
        ("A A", "B B", -110.0, 100.0, "2020-01-10"), ("B B", "A A", 105.0, -125.0, "2020-01-11")
    )
    rows, rep = run(t, s)
    assert rows == [] and len(rep.ambiguous) == 2


def test_two_candidate_fights_stay_unmatched():
    t = tables(
        [("f1", "2020-01-10", "A A", "B B", "win"), ("f2", "2020-01-11", "A A", "B B", "win")]
    )
    rows, rep = run(t, src(("A A", "B B", -110.0, 100.0, "2020-01-10")))
    assert rows == [] and len(rep.ambiguous) == 1


# --- validation -------------------------------------------------------------------------------


def test_overround_out_of_range_is_dropped_and_counted():
    t = tables(
        [("f1", "2020-01-01", "A A", "B B", "win"), ("f2", "2020-01-01", "C C", "D D", "win")]
    )
    s = src(
        ("A A", "B B", -400.0, -400.0, "2020-01-01"), ("C C", "D D", -110.0, -110.0, "2020-01-01")
    )
    rows, rep = run(t, s)
    df = O.make_odds_table(rows, rep)
    assert list(df.fight_id) == ["f2"] and len(rep.dropped_overround) == 1
    O.validate_odds(df, t["fights"])


def test_validate_odds_rejects_bad_tables():
    t = tables([("f1", "2020-01-01", "A A", "B B", "win")])
    good = pd.DataFrame(
        {"fight_id": ["f1"], "odds_f1": [-110.0], "odds_f2": [100.0],
         "p_market_f1": [0.5], "overround": [1.02]}
    )  # fmt: skip
    O.validate_odds(good, t["fights"])
    for bad in (
        pd.concat([good, good]),
        good.assign(fight_id="nope"),
        good.assign(p_market_f1=1.0),
        good.assign(overround=1.4),
    ):
        with pytest.raises(OddsError):
            O.validate_odds(bad, t["fights"])


# --- build and coverage -----------------------------------------------------------------------


def test_build_odds_reports_coverage(tmp_path):
    t = tables([
        ("f1", "2015-03-01", "A A", "B B", "win"),
        ("f2", "2015-03-01", "C C", "D D", "win"),
        ("f3", "2015-03-01", "E E", "F F", "draw"),
        ("f4", "2014-03-01", "G G", "H H", "win"),
    ])  # fmt: skip
    snap = tmp_path / SHA
    snap.mkdir()
    (snap / "ufc-master.csv").write_text(
        "R_fighter,B_fighter,R_odds,B_odds,date,Winner\n"
        "A A,B B,-150,130,2015-03-01,Red\n"
        "X X,Y Y,-150,130,2015-03-01,Red\n"
        "C C,D D,,,2015-03-01,Red\n"
    )
    odds, rep = O.build_odds(snap, t)
    assert list(odds.fight_id) == ["f1"]
    assert rep.per_year["2015"] == {
        "source": 3,
        "with_odds": 2,
        "matched": 1,
        "dropped": 0,
        "unmatched": 1,
    }
    assert (rep.scored_fights, rep.scored_with_odds) == (2, 1)  # f1, f2 are decided from 2015
    assert rep.unmatched == ["2015-03-01 X X vs Y Y"]


# --- fetch and CLI ----------------------------------------------------------------------------


def test_fetch_odds_pins_sha_and_caches(tmp_path):
    import httpx

    from ufc import fetch as F

    calls = []

    def handler(request):
        calls.append(request)
        if request.url.host == "api.github.com":
            assert F.ODDS_REPO in request.url.path
            return httpx.Response(200, text=SHA + "\n")
        assert request.url.path == f"/{F.ODDS_REPO}/{SHA}/{F.ODDS_FILE}"
        return httpx.Response(200, content=(FIX / "odds" / "ufc-master.csv").read_bytes())

    client = httpx.Client(transport=httpx.MockTransport(handler))
    snap = F.fetch_odds(tmp_path, client=client)
    assert snap == tmp_path / "odds" / SHA and (snap / F.ODDS_FILE).exists()
    assert F.latest_odds_snapshot(tmp_path) == snap
    n = len(calls)
    F.fetch_odds(tmp_path, sha=SHA, client=client)  # cached: no download
    assert len(calls) == n


def test_cli_build_writes_odds(tmp_path, capsys):
    raw = tmp_path / "raw"
    shutil.copytree(FIX / "source", raw / SHA)
    (raw / "latest.txt").write_text(SHA)
    shutil.copytree(FIX / "odds", raw / "odds" / SHA)
    (raw / "odds" / "latest.txt").write_text(SHA)
    ov = FIX / "overrides"
    assert main(["--data-dir", str(tmp_path), "--overrides", str(ov), "build"]) == 0
    out = capsys.readouterr().out
    assert "odds: 2 rows" in out and "unmatched odds rows: 1" in out
    assert "Nobody One vs Nobody Two" in out
    odds = pd.read_parquet(tmp_path / "processed" / "odds.parquet")
    assert list(odds.columns) == O.OUT_COLUMNS and len(odds) == 2
    t = {
        n: pd.read_parquet(tmp_path / "processed" / f"{n}.parquet") for n in ("fights", "fighters")
    }
    name = dict(zip(t["fighters"].fighter_id, t["fighters"].name, strict=True))
    f = t["fights"].set_index("fight_id").loc["f000000000000002"]
    row = odds.set_index("fight_id").loc["f000000000000002"]
    price = {name[f.fighter_1_id]: row.odds_f1, name[f.fighter_2_id]: row.odds_f2}
    assert price == {"Dan Dole": 200.0, "Carl Cruz": -250.0}  # odds file lists Dan Dole as red
    assert main(["--data-dir", str(tmp_path), "validate"]) == 0
    assert "odds validation passed" in capsys.readouterr().out


def test_cli_build_without_odds_snapshot_still_builds(tmp_path, capsys):
    raw = tmp_path / "raw"
    shutil.copytree(FIX / "source", raw / SHA)
    (raw / "latest.txt").write_text(SHA)
    assert main(["--data-dir", str(tmp_path), "--overrides", str(FIX / "overrides"), "build"]) == 0
    assert "no odds snapshot" in capsys.readouterr().out
    assert not (tmp_path / "processed" / "odds.parquet").exists()
