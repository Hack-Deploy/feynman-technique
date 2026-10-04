"""Peptide-HLA venue: the judged measurement must be unreachable, and credits must add up."""

from pathlib import Path

import pytest

import protein_market as pm


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "data" / "protein" / "track3_raw.csv").exists(),
    reason="run: uv run python data/protein/make_claims.py",
)


@pytest.fixture(scope="module")
def first_claim():
    return pm.claims()[0]


@pytest.fixture(scope="module")
def lab(first_claim):
    return pm.Lab.for_claim(first_claim)


def test_the_judged_peptide_is_removed_from_the_lab(first_claim, lab):
    assert first_claim["peptide"] not in lab.rows


def test_looking_up_the_judged_peptide_returns_nothing(first_claim, lab):
    found, missing = lab.lookup([first_claim["peptide"]])
    assert found == {}
    assert any("withheld" in m for m in missing)


def test_lookups_are_capped_per_round(lab):
    found, _ = lab.lookup(list(lab.rows)[: pm.MAX_LOOKUPS_PER_ROUND + 10])
    assert len(found) <= pm.MAX_LOOKUPS_PER_ROUND


def test_public_claim_hides_the_answer(first_claim):
    public = pm.public_claim(first_claim)
    assert "answer" not in public and "measured_thalf_hours" not in public


def test_prompt_never_reveals_the_measurement(first_claim, lab):
    att = pm.Attempt(claim=first_claim, model="test", scripted=True)
    text = pm.SYSTEM + pm._prompt(att, lab, first=True)
    assert str(first_claim["measured_thalf_hours"]) not in text


def test_prompt_is_symmetric_between_the_two_verdicts(first_claim, lab):
    """Both verdict words appear in the instructions; neither may appear more often,
    which would hint at the answer."""
    att = pm.Attempt(claim=first_claim, model="test", scripted=True)
    text = (pm.SYSTEM + pm._prompt(att, lab, first=True)).lower()
    assert text.count("supported") == text.count("refuted")


def test_bought_measurements_never_include_the_judged_peptide(first_claim):
    out = pm.run(first_claim["id"], scripted=True)
    bought = {p for r in out["rounds"] for p in (r.get("bought") or {})}
    assert first_claim["peptide"] not in bought


def test_credits_balance_against_spend_and_prize():
    out = pm.run(pm.claims()[0]["id"], scripted=True)
    r = out["result"]
    assert r["balance"] == pytest.approx(pm.BUDGET - r["spent"] + r["prize"])
    assert r["net"] == pytest.approx(r["prize"] - r["spent"])


def test_a_correct_verdict_is_paid_and_a_wrong_one_is_not():
    results = {c["difficulty"]: pm.run(c["id"], scripted=True)["result"] for c in pm.claims()}
    assert results["easy"]["correct"] and results["easy"]["net"] > 0
    assert results["hard"]["correct"] is False and results["hard"]["net"] < 0


def test_base_rate_solver_loses_on_the_hard_claims():
    """The free scripted solver guesses the allele majority; the hard claims punish that."""
    scored = [(c["difficulty"], pm.run(c["id"], scripted=True)["result"]["correct"])
              for c in pm.claims()]
    assert all(ok for d, ok in scored if d != "hard")
    assert not any(ok for d, ok in scored if d == "hard")


def test_live_runs_are_blocked_without_the_env_gates(monkeypatch):
    monkeypatch.delenv("ENABLE_LIVE", raising=False)
    with pytest.raises(RuntimeError, match="ENABLE_LIVE"):
        pm.run(pm.claims()[0]["id"], scripted=False)


@pytest.mark.parametrize("reply,expected", [
    ("<confidence>0.8</confidence><verdict>supported</verdict>", ("supported", 0.8)),
    ("<confidence>1.9</confidence><verdict>refuted</verdict>", ("refuted", 1.0)),
    ("<confidence>abc</confidence><verdict>maybe</verdict>", (None, None)),
])
def test_reply_parsing(reply, expected):
    step = pm._parse(reply)
    assert (step["verdict"], step["confidence"]) == expected
