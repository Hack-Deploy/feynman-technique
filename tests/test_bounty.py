"""The bid rule is applied in code, not by the model."""

import bounty


def _design(p, bsl=1, items=None):
    return {**bounty.EXAMPLE_DESIGN, "p_clear_answer": p, "biosafety_level": bsl,
            "line_items": items or bounty.EXAMPLE_DESIGN["line_items"]}


def test_example_cost_comes_from_price_list():
    k = bounty.decide(bounty.EXAMPLE_DESIGN, 500)
    # 1 plate 6 + 12 reader hours 18 + 2 technician slots 12, as on index.html
    assert k["cost"] == 36.0
    assert k["break_even"] == 36 / 500


def test_bids_only_when_expected_prize_beats_cost():
    assert bounty.decide(_design(0.85), 500)["bids"]
    assert not bounty.decide(_design(0.05), 500)["bids"]  # 25 < 36
    assert not bounty.decide(_design(0.85), 40)["bids"]   # 34 < 36


def test_no_bid_above_bsl2_whatever_the_odds():
    k = bounty.decide(_design(0.99, bsl=3), 10_000)
    assert not k["safe"] and not k["bids"]


def test_stated_probability_is_clamped():
    assert bounty.decide(_design(1.7), 500)["p"] == 1.0
    assert bounty.decide(_design(-0.2), 500)["p"] == 0.0


def test_without_key_returns_labelled_example(monkeypatch):
    monkeypatch.setattr(bounty, "api_key", lambda: None)
    r = bounty.post_bounty("anything", "anything", 999)
    assert r["ok"] and r["source"] == "example"
    assert r["request"] == bounty.EXAMPLE_REQUEST


def test_rejects_empty_form():
    assert not bounty.post_bounty(" ", "x", 100)["ok"]
    assert not bounty.post_bounty("x", "x", 0)["ok"]
