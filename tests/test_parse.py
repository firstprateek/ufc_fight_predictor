import pytest

from ufc import parse as P


def test_null_cells():
    assert P.clean("--") is None and P.clean("---") is None and P.clean("  ") is None
    assert P.parse_of("---") == (None, None)
    assert P.parse_clock("--") is None
    assert P.parse_height_cm("--") is None and P.parse_reach_cm("--") is None


def test_of_and_clock():
    assert P.parse_of("41 of 65") == (41, 65)
    assert P.parse_clock("1:29") == 89 and P.parse_clock("0:00") == 0
    with pytest.raises(ValueError):
        P.parse_of("41/65")


def test_units():
    assert P.parse_height_cm("5' 11\"") == 180.3
    assert P.parse_reach_cm('70"') == 177.8
    assert P.parse_weight_lbs("155 lbs.") == 155
    assert P.parse_date("November 14, 2026") == "2026-11-14"
    assert P.parse_date("Jul 03, 1983") == "1983-07-03"
    assert P.to_int("1.0") == 1 and P.to_int("--") is None


def test_outcome_and_rounds():
    assert P.parse_outcome("W/L") == "first_won"
    assert P.parse_outcome("L/W") == "second_won"
    assert P.parse_outcome("D/D") == "draw" and P.parse_outcome("NC/NC") == "nc"
    assert P.parse_scheduled_rounds("3 Rnd (5-5-5)") == 3
    assert P.parse_scheduled_rounds("5 Rnd (5-5-5-5-5)") == 5
    assert P.parse_scheduled_rounds("1 Rnd + OT (12-3)") == 1
    assert P.parse_scheduled_rounds("No Time Limit") is None
    assert P.parse_round("Round 4") == 4


def test_weight_class():
    wc = P.parse_weight_class("UFC Women's Bantamweight Title Bout")
    assert wc == {"weight_class": "Women's Bantamweight", "is_title": True, "gender": "female"}
    wc = P.parse_weight_class("Light Heavyweight Bout")
    assert wc == {"weight_class": "Light Heavyweight", "is_title": False, "gender": "male"}
    assert P.parse_weight_class("Catch Weight Bout")["weight_class"] == "Catch Weight"
    assert P.parse_weight_class("Road to UFC 3 Flyweight Tournament TitleBout")["is_title"]


def test_fighter_id():
    assert (
        P.fighter_id("http://ufcstats.com/fighter-details/93fe7332d16c6ad9") == "93fe7332d16c6ad9"
    )
