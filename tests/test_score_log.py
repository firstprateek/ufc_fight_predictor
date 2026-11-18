import csv
import json
import math
import re
from datetime import date

import pandas as pd
import pytest

from ufc import score_log as S
from ufc.cli import main
from ufc.predict import LOG_COLUMNS

IDS = list("ABCDEFGHIJ")


def fight(fid, day, a, b, winner, event, outcome="win"):
    return {
        "fight_id": fid,
        "event_id": event,
        "date": day,
        "bout_order": 1,
        "fighter_1_id": min(a, b),
        "fighter_2_id": max(a, b),
        "winner_id": winner,
        "outcome": outcome,
    }


FIGHTS = pd.DataFrame(
    [
        fight("f1", "2026-01-10", "A", "B", "A", "e1"),
        fight("f2", "2026-01-10", "C", "D", "D", "e1"),
        fight("f4", "2026-02-01", "E", "F", None, "e2", "draw"),
        fight("f5", "2026-02-01", "G", "H", "G", "e2"),
        fight("f3", "2026-03-01", "A", "B", "B", "e3"),  # rematch of f1
    ]
)  # the latest fight in the data is 2026-03-01


def row(a="A", b="B", run="2026-01-08T12:00:00Z", day="2026-01-10", p=0.8, odds=("", ""), **kw):
    r = {
        "run_utc": run,
        "data_snapshot": "snap",
        "code_commit": "c1",
        "fighter_1_id": a,
        "fighter_1_name": f"Name {a}",
        "fighter_2_id": b,
        "fighter_2_name": f"Name {b}",
        "date": day,
        "weight_class": "Lightweight",
        "scheduled_rounds": 3,
        "is_title": False,
        "p_blend_1": p,
        "p_hgb_1": p,
        "p_logreg_1": p,
        "odds_1": odds[0],
        "odds_2": odds[1],
    }
    r.update(kw)
    return r


def write(path, rows, header=LOG_COLUMNS):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow([r[c] for c in LOG_COLUMNS])
    return path


def scored(rows, tmp_path, fights=FIGHTS):
    log = write(tmp_path / "log.csv", rows)
    return S.score(S.read_log(log, set(IDS)), fights)


def statuses(rows, tmp_path, fights=FIGHTS):
    log = write(tmp_path / "log.csv", rows)
    rs = S.read_log(log, set(IDS))
    return [e["status"] for e in S._classify(rs, fights)]


# ---------------------------------------------------------------- matching


def test_lower_end_is_the_run_date_and_either_order_matches(tmp_path):
    # f1 (A vs B) is on 01-10 and the rematch f3 on 03-01; both rows name 03-01
    on_run_date = row(run="2026-01-10T06:00:00Z", day="2026-03-01")  # f1 is on/after the run: match
    day_after = row(run="2026-01-11T06:00:00Z", day="2026-03-01")  # f1 is before the run: f3
    swapped = row(a="B", b="A", p=0.2, run="2026-02-01T00:00:00Z", day="2026-03-01")
    log = write(tmp_path / "log.csv", [on_run_date, day_after, swapped])
    es = S._classify(S.read_log(log, set(IDS)), FIGHTS)
    assert [e["fight"]["fight_id"] for e in es] == ["f1", "f3", "f3"]
    assert [e["status"] for e in es] == [S.TOO_LATE, S.SUPERSEDED, S.SCORED]


def test_window_upper_edge_exact(tmp_path):
    # G vs H fought once, on 2026-02-01: the logged date may be at most 45 days before it
    edge = row(a="G", b="H", run="2025-12-20T00:00:00Z", day="2025-12-18")  # window ends 02-01
    miss = row(a="G", b="H", run="2025-12-20T00:00:00Z", day="2025-12-17")  # window ends 01-31
    assert statuses([edge], tmp_path) == [S.SCORED]
    assert statuses([miss], tmp_path) == [S.DID_NOT_HAPPEN]
    # a fight before the run is never a match: C vs D happened on 01-10
    assert statuses([row(a="C", b="D", run="2026-01-20T00:00:00Z")], tmp_path) == [S.TOO_LATE]


def test_rematch_matches_first_fight_on_or_after_run(tmp_path):
    before_f1 = row(run="2026-01-09T00:00:00Z", day="2026-01-10")
    after_f1 = row(run="2026-01-20T00:00:00Z", day="2026-03-01", p=0.3)
    s = scored([before_f1, after_f1], tmp_path)
    assert s["scored_fights"] == 2
    won = {f["fight_date"]: f["winner"] for f in s["fights"]}
    assert won == {"2026-01-10": "Name A", "2026-03-01": "Name B"}


def test_logged_on_or_after_the_fight_date_is_too_late_and_changes_nothing(tmp_path):
    ok = row(run="2026-01-09T23:59:59Z")
    same_day = row(run="2026-01-10T00:00:00Z", p=0.99)
    after = row(run="2026-01-11T09:00:00Z", p=0.99)
    base = scored([ok], tmp_path)
    s = scored([ok, same_day, after], tmp_path)
    assert statuses([ok, same_day, after], tmp_path) == [S.SCORED, S.TOO_LATE, S.TOO_LATE]
    assert s["models"] == base["models"] and s["fights"] == base["fights"]
    assert s["status_counts"][S.SCORED] == 1 and s["status_counts"][S.TOO_LATE] == 2


def test_backdated_replay_is_never_scored(tmp_path):
    # written in March about a fight on 01-10 that had already happened
    replay = row(run="2026-03-05T00:00:00Z", day="2026-01-10", p=0.99)
    s = scored([replay], tmp_path)
    assert s["scored_fights"] == 0 and s["models"] is None
    assert statuses([replay], tmp_path) == [S.TOO_LATE]
    assert "after the fight" in s["not_scored"][0]["reason"]


def test_latest_eligible_row_wins_whatever_its_probability(tmp_path):
    good = row(run="2026-01-08T10:00:00Z", p=0.95)  # A won: the better forecast, but earlier
    bad = row(run="2026-01-09T10:00:00Z", p=0.05)
    s = scored([good, bad], tmp_path)
    assert statuses([good, bad], tmp_path) == [S.SUPERSEDED, S.SCORED]
    assert s["status_counts"][S.SUPERSEDED] == 1
    assert s["models"]["blend"]["log_loss"] == pytest.approx(-math.log(0.05))
    # the same in the other file order: the latest run still wins
    assert statuses([bad, good], tmp_path) == [S.SCORED, S.SUPERSEDED]
    assert scored([bad, good], tmp_path)["models"]["blend"]["log_loss"] == pytest.approx(
        -math.log(0.05)
    )


def test_every_status_and_pending_versus_did_not_happen(tmp_path):
    rows = [
        row(),  # scored
        row(run="2026-01-07T00:00:00Z"),  # superseded by the one above
        row(run="2026-01-10T08:00:00Z", a="C", b="D"),  # logged too late
        row(run="2026-01-31T00:00:00Z", day="2026-02-01", a="E", b="F"),  # draw -> no decision
        row(run="2026-02-01T00:00:00Z", day="2026-02-02", a="I", b="J"),  # window open: 03-19
        row(run="2026-01-01T00:00:00Z", day="2026-01-05", a="I", b="J"),  # window over: 02-19
    ]
    assert statuses(rows, tmp_path) == [
        S.SCORED,
        S.SUPERSEDED,
        S.TOO_LATE,
        S.NO_DECISION,
        S.PENDING,
        S.DID_NOT_HAPPEN,
    ]
    s = scored(rows, tmp_path)
    assert all(s["status_counts"][k] == 1 for k in S.STATUSES)
    assert {r["status"] for r in s["not_scored"]} == {S.TOO_LATE, S.NO_DECISION, S.DID_NOT_HAPPEN}
    # the clock is not used: with a later latest fight the pending row becomes did not happen
    later = pd.concat([FIGHTS, pd.DataFrame([fight("f9", "2026-04-01", "G", "H", "H", "e9")])])
    assert statuses(rows, tmp_path, later)[4] == S.DID_NOT_HAPPEN


# ---------------------------------------------------------------- scoring


def test_orientation_gives_the_same_loss(tmp_path):
    a = row(a="A", b="B", p=0.7)
    b = row(a="B", b="A", p=0.3)
    sa, sb = scored([a], tmp_path), scored([b], tmp_path)
    assert sa["models"]["blend"]["log_loss"] == pytest.approx(sb["models"]["blend"]["log_loss"])
    assert sa["models"]["blend"]["log_loss"] == pytest.approx(-math.log(0.7))
    assert sa["models"]["blend"]["brier"] == pytest.approx(sb["models"]["blend"]["brier"])
    assert sa["models"]["blend"]["accuracy"] == sb["models"]["blend"]["accuracy"] == 1.0


def test_hand_checked_metrics_and_market_block(tmp_path):
    rows = [
        row(p=0.8, odds=(-200, 170)),  # A beat B (event e1), has prices
        row(a="D", b="C", p=0.6),  # D beat C (event e1), no prices
        row(a="G", b="H", run="2026-01-30T00:00:00Z", day="2026-02-01", p=0.4, odds=(120, -140)),
    ]  # G beat H (event e2), has prices
    s = scored(rows, tmp_path)
    ll = [-math.log(0.8), -math.log(0.6), -math.log(0.4)]
    br = [0.2**2, 0.4**2, 0.6**2]
    m = s["models"]["blend"]
    assert m["fights"] == 3
    assert m["log_loss"] == pytest.approx(sum(ll) / 3)
    assert m["brier"] == pytest.approx(sum(br) / 3)
    assert m["accuracy"] == pytest.approx(2 / 3)
    assert s["coin_flip"]["log_loss"] == pytest.approx(0.6931, abs=5e-5)
    # market block: only the two rows with both prices
    mk = s["market"]
    assert mk["fights"] == 2 and mk["events"] == 2
    q1, q2 = 200 / 300, 100 / 270
    p_a = q1 / (q1 + q2)
    r1, r2 = 100 / 220, 140 / 240
    p_g = r1 / (r1 + r2)
    assert mk["market"]["log_loss"] == pytest.approx((-math.log(p_a) - math.log(p_g)) / 2)
    assert mk["blend"]["log_loss"] == pytest.approx((ll[0] + ll[2]) / 2)
    assert mk["market"]["accuracy"] == pytest.approx(0.5)  # G was the underdog
    assert mk["blend"]["accuracy"] == pytest.approx(0.5)
    d = mk["difference"]
    assert d["mean"] == pytest.approx(mk["blend"]["log_loss"] - mk["market"]["log_loss"])
    assert d["low"] <= d["mean"] <= d["high"]
    assert [f["p_market_1"] is None for f in s["fights"]] == [False, True, False]
    assert s["fights"][0]["log_loss_blend"] == pytest.approx(ll[0])


def test_no_interval_with_one_event(tmp_path):
    s = scored([row(odds=(-200, 170))], tmp_path)
    assert s["market"]["difference"] is None
    assert "at least 2 events" in s["market"]["note"]
    assert "at least 2 events" in S.render(s)


def test_sample_size_line_appears_below_the_threshold_only():
    def doc(n):
        fights = pd.DataFrame(
            [fight(f"x{i}", "2026-01-10", f"p{i}", f"q{i}", f"p{i}", f"e{i}") for i in range(n)]
        )
        rows = [
            S.Row(
                line=i + 2,
                run=pd.Timestamp("2026-01-01", tz="UTC").to_pydatetime(),
                day=date(2026, 1, 1),
                fight_date=date(2026, 1, 10),
                f1=f"p{i}",
                n1="N",
                f2=f"q{i}",
                n2="M",
                probs={"blend": 0.6, "hgb": 0.6, "logreg": 0.6},
                odds=None,
                commit="c",
                snapshot="s",
            )
            for i in range(n)
        ]
        return S.score(rows, fights)

    below, at = doc(S.MIN_SCORED_FIGHTS - 1), doc(S.MIN_SCORED_FIGHTS)
    assert below["scored_fights"] == 199 and "too few to tell" in S.render(below)
    assert "0.6092" in S.render(below) and "0.6474" in S.render(below)
    assert at["scored_fights"] == 200 and "too few" not in S.render(at)
    assert at["too_few_note"] is None


def test_provenance_counts(tmp_path):
    rows = [
        row(code_commit="abc"),
        row(a="D", b="C", code_commit="abc-dirty", data_snapshot="snap2"),
        row(a="G", b="H", run="2026-01-30T00:00:00Z", day="2026-02-01", code_commit="unknown"),
        row(a="E", b="F", run="2026-01-30T00:00:00Z", day="2026-02-01", code_commit="zzz-dirty"),
    ]  # the last is a draw: not scored, so not counted
    pv = scored(rows, tmp_path)["provenance"]
    assert pv == {"code_commits": 3, "data_snapshots": 2, "dirty_or_unknown_commit": 2}


# ---------------------------------------------------------------- edge cases and the CLI


@pytest.fixture
def env(tmp_path, monkeypatch):
    proc = tmp_path / "data" / "processed"
    proc.mkdir(parents=True)
    FIGHTS.to_parquet(proc / "fights.parquet", index=False)
    pd.DataFrame({"fighter_id": IDS}).to_parquet(proc / "fighters.parquet", index=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path, ["--data-dir", str(tmp_path / "data"), "score-log"]


def test_missing_and_empty_log_exit_zero(env, capsys):
    tmp, base = env
    assert main(base) == 0
    assert "no predictions to score" in capsys.readouterr().out
    (tmp / "predictions").mkdir()
    (tmp / "predictions" / "log.csv").write_text("")
    assert main(base) == 0
    assert "no predictions to score" in capsys.readouterr().out


def test_wrong_header_fails(env, capsys):
    tmp, base = env
    p = write(tmp / "log.csv", [row()], header=LOG_COLUMNS[::-1])
    assert main(base + ["--log", str(p)]) == 1
    assert "header" in capsys.readouterr().err


@pytest.mark.parametrize(
    "bad,what",
    [
        ({"fighter_1_id": "ZZ"}, "unknown fighter id"),
        ({"p_blend_1": 1.0}, "outside (0, 1)"),
        ({"p_hgb_1": 0}, "outside (0, 1)"),
        ({"p_logreg_1": "nan"}, "finite"),
        ({"date": "10/01/2026"}, "cannot read date"),
        ({"run_utc": "yesterday"}, "cannot read run_utc"),
        ({"odds_1": -150}, "only one of"),
    ],
)
def test_a_bad_row_fails_naming_its_line(env, capsys, bad, what):
    tmp, base = env
    p = write(tmp / "log.csv", [row(), row(**bad), row()])
    assert main(base + ["--log", str(p)]) == 1
    err = capsys.readouterr().err
    assert "line 3" in err and what in err


def test_header_only_log_prints_status_counts_and_no_tables(env, capsys):
    tmp, base = env
    p = write(tmp / "log.csv", [])
    assert main(base + ["--log", str(p)]) == 0
    out = capsys.readouterr().out
    assert "Scored fights: 0" in out and "Rows by status" in out and "Models" not in out


def test_zero_scored_rows_have_no_metric_tables(env, capsys):
    tmp, base = env
    p = write(tmp / "log.csv", [row(run="2026-01-10T00:00:00Z")])
    assert main(base + ["--log", str(p)]) == 0
    out = capsys.readouterr().out
    assert "logged too late" in out and "Models" not in out and "Against the market" not in out


def test_run_leaves_the_log_untouched_and_json_and_out_work(env, capsys):
    tmp, base = env
    p = write(
        tmp / "log.csv",
        [row(odds=(-200, 170)), row(a="G", b="H", run="2026-01-30T00:00:00Z", day="2026-02-01")],
    )
    before = p.read_bytes()
    out_file = tmp / "sub" / "report.txt"
    assert main(base + ["--log", str(p), "--out", str(out_file)]) == 0
    text = capsys.readouterr().out
    assert out_file.read_text() == text
    assert "strictly before" in text and "at least the day before" in text
    assert main(base + ["--log", str(p), "--json"]) == 0
    doc = json.loads(capsys.readouterr().out)
    assert doc["scored_fights"] == 2 and doc["status_counts"]["scored"] == 2
    assert p.read_bytes() == before


def test_no_advice_wording(env, capsys):
    tmp, base = env
    p = write(
        tmp / "log.csv",
        [row(odds=(-200, 170)), row(a="G", b="H", run="2026-01-30T00:00:00Z", day="2026-02-01")],
    )
    texts = []
    for extra in ([], ["--json"]):
        main(base + ["--log", str(p)] + extra)
        texts.append(capsys.readouterr().out)
    banned = re.compile(
        r"\b(bet|bets|betting|stake|stakes|profit|profits|value|values|advice|wager)\b", re.I
    )
    for t in texts:
        assert not banned.search(t)
