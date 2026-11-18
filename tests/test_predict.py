import csv
import io
import json
import re
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from test_backtest import synthetic

from ufc import predict as P
from ufc.backtest.run import scored_fights, walk_forward
from ufc.cli import main
from ufc.features import build_features, fit_imputation, parse_feature_set
from ufc.odds import implied_prob, remove_margin

NAMES = {
    "f00": "Alex Alpha",
    "f01": "Bo Beta",
    "f02": "José Núñez",
    "f05": "Ivan Imputed",
    "f10": "Bruno Silva",
    "f11": "Bruno Silva",
    "f22": "Hugo Heavy",
    "f23": "Larry Layoff",
    "f24": "Newcomer Nobody",
}
GROUPS = parse_feature_set("all")  # the frozen feature set
STATS_COLUMNS = ["fight_id", "fighter_id", "round"]


@pytest.fixture(scope="module")
def world():
    fights, fighters = synthetic()
    fighters["name"] = [NAMES.get(f, f"Fighter {f}") for f in fighters.fighter_id]
    fighters = pd.concat(
        [
            fighters,
            pd.DataFrame(
                [
                    {
                        "fighter_id": "f24",
                        "name": NAMES["f24"],
                        "height_cm": 180.0,
                        "reach_cm": 185.0,
                        "stance": "Orthodox",
                        "dob": "1995-05-05",
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    fighters.loc[fighters.fighter_id == "f05", "reach_cm"] = np.nan  # a filled-in snapshot field
    gone = ((fights.fighter_1_id == "f23") | (fights.fighter_2_id == "f23")) & (
        fights.date > "2013-01-01"
    )
    fights = fights[~gone].reset_index(drop=True)
    heavy = (fights.fighter_1_id == "f22") | (fights.fighter_2_id == "f22")
    fights.loc[heavy, "weight_class"] = "Heavyweight"
    fights["scheduled_rounds"] = pd.array([3] * len(fights), dtype="Int64")
    fights["is_title"] = False
    fights["referee"] = None
    stats = pd.DataFrame(columns=STATS_COLUMNS)
    tables = {"fights": fights, "fighters": fighters, "fight_stats_round": stats, "events": None}
    latest = fights.date.max()
    day = (date.fromisoformat(latest) + timedelta(days=3)).isoformat()
    return tables, day


def lightweights(tables, day):
    """Fighters (with names that are unique) whose last fight before `day` was Lightweight."""
    f = tables["fights"]
    f = f[f.date < day]
    out = []
    for fid in tables["fighters"].fighter_id:
        mine = f[(f.fighter_1_id == fid) | (f.fighter_2_id == fid)]
        if len(mine) >= 5 and mine.iloc[-1].weight_class == "Lightweight" and fid not in NAMES:
            out.append(fid)
    return out


def req(tables, a, b, **kw):
    n = tables["fighters"].set_index("fighter_id").name
    return P.Request(n[a], n[b], id1=a, id2=b, **kw)


def one(world, a, b, **kw):
    tables, day = world
    return P.run([req(tables, a, b, **kw)], tables, "snap", default_date=day)["fights"][0]


def synth_snapshot(tables, directory):
    directory.mkdir(parents=True, exist_ok=True)
    for k in ("fights", "fighters", "fight_stats_round"):
        tables[k].to_parquet(directory / f"{k}.parquet", index=False)
    pd.DataFrame({"event_id": ["e"]}).to_parquet(directory / "events.parquet", index=False)


# -------------------------------------------------------------- same code path, point in time


def test_proposed_rows_leave_every_other_row_identical(world):
    tables, day = world
    fights, fighters, stats = tables["fights"], tables["fighters"], tables["fight_stats_round"]
    imp = fit_imputation(fights, fighters)
    mid = fights.date.iloc[len(fights) // 2]  # a date with fights before and after it
    lw = lightweights(tables, day)
    ms = [
        P.Matchup(lw[0], lw[1], "a", "b", mid, "Lightweight", 3, False, "male"),
        P.Matchup(lw[1], lw[2], "b", "c", mid, "Lightweight", 3, False, "male"),
    ]
    base = build_features(fights, stats, fighters, imp)
    both = build_features(P.append_proposed(fights, ms), stats, fighters, imp)
    real = both[~both.fight_id.str.startswith(P.PROPOSED)].reset_index(drop=True)
    pd.testing.assert_frame_equal(real, base, check_exact=True)
    buf = [io.BytesIO(), io.BytesIO()]
    real.to_parquet(buf[0])
    base.to_parquet(buf[1])
    assert buf[0].getvalue() == buf[1].getvalue()
    assert len(both) == len(base) + 2  # and the later fights were all still featurised


def test_proposed_row_has_no_result_and_is_skipped_by_every_state(world):
    tables, day = world
    f = P.append_proposed(
        tables["fights"], [P.Matchup("f01", "f00", "b", "a", day, "Lightweight", 3, False, "male")]
    )
    new = f.iloc[-1]
    assert pd.isna(new.outcome) and pd.isna(new.winner_id)
    assert (new.fighter_1_id, new.fighter_2_id) == ("f00", "f01")  # stored order, ids sorted


def test_features_only_see_fights_before_the_date(world):
    tables, _ = world
    f = tables["fights"]
    cut = f.date.iloc[len(f) // 2]
    lw = lightweights(tables, cut)
    a, b = lw[0], lw[1]
    r = P.run([req(tables, a, b)], tables, "snap", default_date=cut)["fights"][0]
    before = f[f.date < cut]
    n = ((before.fighter_1_id == a) | (before.fighter_2_id == a)).sum()
    assert r["fighter_1"]["ufc_fights"] == n
    assert r["fights_trained_on"] <= len(before)
    # a date before most of the data is allowed and simply ignores the later fights
    early = f.date.iloc[40]
    r2 = P.run([req(tables, "f00", "f01", weight_class="Lightweight")], tables, "s", early)
    assert r2["fights"][0]["fights_trained_on"] < r["fights_trained_on"]


def test_replay_reproduces_the_walk_forward_prediction(world):
    tables, _ = world
    fights, fighters, stats = tables["fights"], tables["fighters"], tables["fight_stats_round"]
    feats = build_features(fights, stats, fighters)
    pred = walk_forward(scored_fights(feats, fights), "2015-01-01", groups=GROUPS)
    fx = fights.set_index("fight_id")
    for i in (0, len(pred) // 3, len(pred) // 2, len(pred) - 1):
        row = pred.iloc[i]
        x = fx.loc[row.fight_id]
        r = req(tables, x.fighter_1_id, x.fighter_2_id, date=x.date, weight_class=x.weight_class)
        m = P.resolve(r, tables, "2099-01-01")
        got = P.predict_matchups([m], tables, train_before=row.quarter)[0]
        for name in ("hgb", "logreg", "blend"):
            assert abs(got["probabilities_fighter_1"][name] - row[f"p_{name}"]) < 1e-9, name
        assert got["training_cutoff"] == row.quarter


def test_training_cutoff_cannot_be_after_the_date(world):
    tables, day = world
    m = P.resolve(req(tables, "f00", "f01", weight_class="Lightweight"), tables, day)
    with pytest.raises(P.PredictError, match="after the fight date"):
        P.predict_matchups([m], tables, train_before="2099-01-01")


def test_swapping_the_names_mirrors_the_result(world):
    a, b = lightweights(*world)[:2]
    x, y = one(world, a, b), one(world, b, a)
    for k in ("hgb", "logreg", "blend"):
        p, q = x["probabilities_fighter_1"][k], y["probabilities_fighter_1"][k]
        assert p + q == pytest.approx(1.0, abs=1e-12)
    assert x["fighter_1"]["name"] == y["fighter_2"]["name"]
    assert x["fighter_1"]["p_win"] == pytest.approx(y["fighter_2"]["p_win"], abs=1e-12)
    assert x["fighter_1"]["elo"] == y["fighter_2"]["elo"]
    assert x["fighter_1"]["p_win"] + x["fighter_2"]["p_win"] == pytest.approx(1.0)


def test_card_fights_on_one_date_do_not_see_each_other(world):
    tables, day = world
    a, b, c = lightweights(*world)[:3]
    card = [req(tables, a, b), req(tables, b, c)]
    both = P.run(card, tables, "s", default_date=day)["fights"]
    alone = [P.run([r], tables, "s", default_date=day)["fights"][0] for r in card]
    for got, want in zip(both, alone, strict=True):
        assert got["probabilities_fighter_1"] == pytest.approx(want["probabilities_fighter_1"])
        for side in ("fighter_1", "fighter_2"):
            for k in ("ufc_fights", "ufc_record", "elo", "glicko", "days_since_last_fight"):
                assert got[side][k] == want[side][k]
    past = tables["fights"][tables["fights"].date < day]
    n_b = ((past.fighter_1_id == b) | (past.fighter_2_id == b)).sum()
    assert both[0]["fighter_2"]["ufc_fights"] == both[1]["fighter_1"]["ufc_fights"] == n_b


# -------------------------------------------------------------- names


def test_names_ignore_case_accents_and_spaces(world):
    tables, day = world
    for typed in ("jose nunez", "  JOSE   NUNEZ ", "José Núñez", "JOSÉ NÚÑEZ"):
        m = P.resolve(P.Request(typed, "Bo Beta", weight_class="Lightweight"), tables, day)
        assert (m.id1, m.name1) == ("f02", "José Núñez")


def test_unknown_name_fails_with_close_names_and_a_refresh_hint(world):
    tables, day = world
    with pytest.raises(P.PredictError) as e:
        P.resolve(P.Request("Jose Nunes", "Bo Beta"), tables, day)
    msg = str(e.value)
    assert "José Núñez" in msg and "ufc fetch && ufc build" in msg
    with pytest.raises(P.PredictError, match="no close names"):
        P.resolve(P.Request("Qqqqqqqq Zzzzzzz", "Bo Beta"), tables, day)


def test_shared_name_is_refused_listing_candidates_and_resolved_by_id(world):
    tables, day = world
    with pytest.raises(P.PredictError) as e:
        P.resolve(P.Request("bruno silva", "Bo Beta", weight_class="Lightweight"), tables, day)
    msg = str(e.value)
    assert "f10" in msg and "f11" in msg and "--id-1" in msg
    assert "UFC record" in msg and "last fight 20" in msg and "Lightweight" in msg
    m = P.resolve(
        P.Request("bruno silva", "Bo Beta", id1="f11", weight_class="Lightweight"), tables, day
    )
    assert m.id1 == "f11"
    with pytest.raises(P.PredictError, match="--id-2"):
        P.resolve(P.Request("Bo Beta", "Bruno Silva", weight_class="Lightweight"), tables, day)
    with pytest.raises(P.PredictError, match="not 'Bo Beta'"):  # an id for someone else
        P.resolve(P.Request("Bo Beta", "Alex Alpha", id1="f00"), tables, day)
    with pytest.raises(P.PredictError, match="no fighter has the id"):
        P.resolve(P.Request("Bo Beta", "Alex Alpha", id1="nope"), tables, day)


def test_the_same_fighter_on_both_sides_is_an_error(world):
    tables, day = world
    with pytest.raises(P.PredictError, match="both sides"):
        P.resolve(P.Request("Bo Beta", "bo  BETA"), tables, day)


# -------------------------------------------------------------- defaults


def test_weight_class_defaults_to_the_class_both_last_fought_in(world):
    tables, day = world
    a, b = lightweights(*world)[:2]
    m = P.resolve(req(tables, a, b), tables, day)
    assert (m.weight_class, m.rounds, m.title, m.gender) == ("Lightweight", 3, False, "male")


def test_weight_class_is_required_when_the_classes_differ_or_a_fighter_is_new(world):
    tables, day = world
    a = lightweights(*world)[0]
    with pytest.raises(P.PredictError, match=r"Hugo Heavy Heavyweight.*pass --weight-class"):
        P.resolve(req(tables, a, "f22"), tables, day)
    with pytest.raises(P.PredictError, match="Newcomer Nobody no UFC fight"):
        P.resolve(req(tables, a, "f24"), tables, day)
    ok = P.resolve(req(tables, a, "f22", weight_class="lightweight"), tables, day)
    assert ok.weight_class == "Lightweight"
    with pytest.raises(P.PredictError, match="unknown weight class"):
        P.resolve(req(tables, a, "f22", weight_class="Sumo"), tables, day)


def test_title_implies_five_rounds(world):
    tables, day = world
    a, b = lightweights(*world)[:2]
    m = P.resolve(req(tables, a, b, title=True), tables, day)
    assert (m.title, m.rounds) == (True, 5)
    assert P.resolve(req(tables, a, b, rounds=5), tables, day).rounds == 5
    with pytest.raises(P.PredictError, match="5 rounds"):
        P.resolve(req(tables, a, b, title=True, rounds=3), tables, day)


# -------------------------------------------------------------- notes and warnings


def test_notes_for_debut_few_fights_imputed_and_layoff(world):
    tables, day = world
    r = one(world, "f24", "f05", weight_class="Lightweight")
    text = " ".join(r["notes"])
    assert "UFC debut: Newcomer Nobody" in text
    assert "Fewer than 3 prior UFC fights: Newcomer Nobody (0)" in text
    assert "Physical data filled in" in text and "Ivan Imputed" in text
    assert "Newcomer" not in text.split("Physical data filled in")[1]
    assert r["fighter_1"]["last_fight_date"] is None
    rl = one(world, "f23", "f00", weight_class="Lightweight")
    assert any(n.startswith("Layoff over 730 days: Larry Layoff") for n in rl["notes"])
    a, b = lightweights(*world)[:2]
    assert one(world, a, b)["notes"] == []


def test_stale_data_warning(world):
    tables, day = world
    latest = tables["fights"].date.max()
    far = (date.fromisoformat(latest) + timedelta(days=15)).isoformat()
    near = (date.fromisoformat(latest) + timedelta(days=14)).isoformat()
    r = req(tables, "f00", "f01", weight_class="Lightweight")
    assert "15 days before" in P.run([r], tables, "s", far)["warnings"][0]
    assert P.run([r], tables, "s", near)["warnings"] == []
    doc = P.run([r], tables, "snapsha", far)
    assert "ufc fetch && ufc build" in P.render(doc)
    assert "WARNING" in P.render(doc) and "snapsha" in P.render(doc) and latest in P.render(doc)


# -------------------------------------------------------------- odds


def test_odds_block_removes_the_margin_and_gives_no_advice(world):
    tables, day = world
    a, b = lightweights(*world)[:2]
    doc = P.run([req(tables, a, b, odds=(-150, 130))], tables, "s", day)
    m = doc["fights"][0]["market"]
    q1, q2 = implied_prob(-150), implied_prob(130)
    assert m["p_market_1"] == pytest.approx(q1 / (q1 + q2))
    assert m["p_market_1"] + m["p_market_2"] == pytest.approx(1.0)
    assert m["overround"] == pytest.approx(q1 + q2) and m["overround"] > 1
    assert m["difference_1"] == pytest.approx(m["p_model_1"] - m["p_market_1"])
    assert remove_margin(-150, 130)[0] == pytest.approx(m["p_market_1"])
    text = P.render(doc)
    assert P.MARKET_NOTE in text
    assert "5,179" in text and "0.6092" in text and "0.6474" in text
    assert "market" in text and "difference" in text
    banned = r"\b(bet|bets|betting|stake|value|edge|kelly|wager|bankroll|recommend\w*|should)\b"
    assert not re.search(banned, text, re.IGNORECASE)
    assert P.render(P.run([req(tables, a, b)], tables, "s", day)).count("Market (") == 0


@pytest.mark.parametrize("odds", [(-50, 130), (150, 130), (-1000, -1000), (-110, 5)])
def test_bad_odds_are_refused(world, odds):
    tables, day = world
    with pytest.raises(P.PredictError):
        P.resolve(req(tables, "f00", "f01", weight_class="Lightweight", odds=odds), tables, day)


# -------------------------------------------------------------- output and log


def test_json_parses_and_carries_every_probability(world):
    tables, day = world
    a, b = lightweights(*world)[:2]
    doc = json.loads(P.to_json(P.run([req(tables, a, b, odds=(-120, 100))], tables, "s", day)))
    f = doc["fights"][0]
    assert set(f["probabilities_fighter_1"]) == {"blend", "hgb", "logreg"}
    assert f["fighter_1"]["p_win"] == f["probabilities_fighter_1"]["blend"]
    assert doc["data"]["snapshot"] == "s" and doc["data"]["fights_trained_on"]
    assert f["market"]["odds_1"] == -120 and f["fighter_1"]["fair_american_odds"]


def test_fair_american_odds():
    assert P.fair_american(0.5) == "+100"
    assert P.fair_american(0.75) == "-300"
    assert P.fair_american(0.25) == "+300"
    assert P.fair_american(0.6) == "-150"


def test_log_appends_and_never_rewrites(world, tmp_path):
    tables, day = world
    a, b, c = lightweights(*world)[:3]
    path = tmp_path / "predictions" / "log.csv"
    d1 = P.run([req(tables, a, b, odds=(-150, 130)), req(tables, b, c)], tables, "snap1", day)
    assert P.append_log(path, d1, commit="abc123") == 2
    first = path.read_bytes()
    d2 = P.run([req(tables, c, a, title=True)], tables, "snap2", day)
    P.append_log(path, d2, commit="def456")
    assert path.read_bytes().startswith(first)
    rows = list(csv.DictReader(path.open()))
    assert list(rows[0]) == P.LOG_COLUMNS and len(rows) == 3
    assert rows[0]["fighter_1_id"] == a and rows[0]["odds_1"] == "-150" and rows[1]["odds_1"] == ""
    assert rows[0]["data_snapshot"] == "snap1" and rows[2]["code_commit"] == "def456"
    assert rows[2]["is_title"] == "True" and rows[2]["scheduled_rounds"] == "5"
    assert float(rows[0]["p_blend_1"]) == pytest.approx(d1["fights"][0]["fighter_1"]["p_win"])
    assert rows[0]["run_utc"].endswith("Z")
    path.write_text("a,b\n1,2\n")
    with pytest.raises(P.PredictError, match="different header"):
        P.append_log(path, d1, commit="x")
    assert path.read_text() == "a,b\n1,2\n"


# -------------------------------------------------------------- card and CLI


def test_card_file_parses_defaults_and_rejects_mistakes(tmp_path):
    f = tmp_path / "c.csv"
    f.write_text(
        "fighter_1,fighter_2,date,scheduled_rounds,is_title,odds_1,odds_2\n"
        "A,B,2026-01-01,5,true,-150,+130\nC,D,,,,,\n"
    )
    r = P.load_card(f)
    assert r[0] == P.Request("A", "B", None, None, "2026-01-01", None, 5, True, (-150.0, 130.0))
    assert r[1] == P.Request("C", "D")
    for text, msg in (
        ("fighter_1\nA\n", "needs the columns"),
        ("fighter_1,fighter_2,colour\nA,B,red\n", "unknown card columns"),
        ("fighter_1,fighter_2,odds_1\nA,B,-150\n", "both odds_1 and odds_2"),
        ("fighter_1,fighter_2\nA,\n", "both fighter names"),
        ("fighter_1,fighter_2\n", "no fights"),
        ("fighter_1,fighter_2,is_title\nA,B,maybe\n", "true or false"),
    ):
        f.write_text(text)
        with pytest.raises(P.PredictError, match=msg):
            P.load_card(f)


def test_the_committed_example_card_parses():
    cards = P.load_card(__import__("pathlib").Path("examples/card.csv"))
    assert len(cards) >= 4 and all(c.name1 and c.name2 for c in cards)


def test_cli_single_fight_card_json_and_log(world, tmp_path, monkeypatch, capsys):
    tables, day = world
    synth_snapshot(tables, tmp_path / "data" / "processed")
    monkeypatch.chdir(tmp_path)
    base = ["--data-dir", str(tmp_path / "data"), "predict"]
    a, b = lightweights(*world)[:2]
    n = tables["fighters"].set_index("fighter_id").name
    args = [n[a], n[b], "--date", day, "--odds", "-150", "130", "--log"]
    assert main(base + args) == 0
    out = capsys.readouterr()
    assert f"{n[a]} vs {n[b]}" in out.out and "Market (-150 / +130" in out.out
    assert "logged 1 fight" in out.err
    assert main(base + args + ["--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["fights"][0]["market"]["odds_2"] == 130
    assert len(list(csv.DictReader(open("predictions/log.csv")))) == 2
    card = tmp_path / "card.csv"
    card.write_text(f"fighter_1,fighter_2\n{n[a]},{n[b]}\n")
    assert main(base + ["--card", str(card), "--date", day]) == 0
    assert capsys.readouterr().out.count(" vs ") == 1
    # errors exit non-zero with a message and never a traceback
    assert main(base + ["Nobody Atall", n[b], "--date", day]) == 1
    assert "ufc fetch && ufc build" in capsys.readouterr().err
    assert main(base + [n[a]]) == 1
    assert main(base + ["--card", str(card), n[a]]) == 1
    assert main(base + [n[a], n[b], "--date", "01/02/2026"]) == 1
    assert "YYYY-MM-DD" in capsys.readouterr().err
