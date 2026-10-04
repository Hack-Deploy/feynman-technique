"""Every protein claim must match the published measurement it is judged against."""

import json
from pathlib import Path

import pandas as pd
import pytest

DATA = Path(__file__).resolve().parent.parent / "data" / "protein"
THRESHOLD_H = 1.0
MARGIN = 5.0


@pytest.fixture(scope="module")
def claims():
    return json.loads((DATA / "claims.json").read_text())


pytestmark = pytest.mark.skipif(
    not (Path(__file__).resolve().parent.parent / "data" / "protein" / "track3_raw.csv").exists(),
    reason="run: uv run python data/protein/make_claims.py",
)

@pytest.fixture(scope="module")
def raw():
    return pd.read_csv(DATA / "track3_raw.csv")


def test_measurement_matches_the_dataset(claims, raw):
    for c in claims:
        rows = raw[(raw.allele == c["allele"]) & (raw.peptide == c["peptide"])]
        assert len(rows) == 1, f"{c['peptide']}/{c['allele']}: {len(rows)} rows, expected 1"
        assert rows.iloc[0].thalf_hours == pytest.approx(c["measured_thalf_hours"])


def test_verdict_follows_from_the_measurement(claims):
    for c in claims:
        above = c["measured_thalf_hours"] > THRESHOLD_H
        assert c["answer"] == ("supported" if above else "refuted")


def test_no_claim_sits_within_measurement_noise(claims):
    """A 5x margin from the threshold, so no verdict turns on a borderline reading."""
    for c in claims:
        t = c["measured_thalf_hours"]
        assert t >= THRESHOLD_H * MARGIN or t <= THRESHOLD_H / MARGIN, f"{c['peptide']}: {t} h"


def test_half_the_claims_are_false(claims):
    n_true = sum(c["answer"] == "supported" for c in claims)
    assert n_true * 2 == len(claims), "guessing one answer every time must score 50%"


def test_some_claims_punish_base_rate_guessing(claims):
    against = [c for c in claims if c["base_rate_answer"] != c["answer"]]
    assert len(against) >= 2, "at least two claims where the allele's majority answer is wrong"


def test_claim_ids_are_unique(claims):
    ids = [c["id"] for c in claims]
    assert len(set(ids)) == len(ids)
