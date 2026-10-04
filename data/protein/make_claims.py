"""Pick six peptide-HLA claims from the Track 3 stability dataset.

Each claim asks whether one measured binding half-life is above 1 hour. The
measurement is the hidden answer; the AI scientist never sees that row.

Design rules:
  * three true, three false, so guessing one answer every time scores 50%
  * a 5x margin from the 1-hour line, so no claim turns on measurement noise
  * difficulty comes from the allele's base rate, not from a borderline value:
    a "hard" claim is one where the allele's majority answer is wrong
  * deterministic: rows are sorted before picking, so this reproduces exactly

Run: uv run python data/protein/make_claims.py
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "track3_raw.csv"
SOURCE = ("https://docs.google.com/spreadsheets/d/"
          "1NtZNvcF3u0KFn-1bbuA50CF3IXvs1l4HfbaR3KRLvso/export?format=csv")
THRESHOLD_H = 1.0
MARGIN = 5.0  # true claims >= 5 h, false claims <= 0.2 h

# (allele, answer, difficulty). Hard = the allele's base rate points the wrong way.
PICKS = [
    ("HLA-B*15:01", True, "easy"),             # 89% of this allele's peptides are above
    ("HLA-A*02:01", True, "medium"),           # 72% above
    ("HLA-B*14:01(C67S)", True, "hard"),       # only 3% above: the base rate says refuted
    ("HLA-B*08:01", False, "easy"),            # only 8% above
    ("HLA-A*03:01", False, "medium"),          # 47% above: the base rate says nothing
    ("HLA-A*02:11", False, "hard"),            # 92% above: the base rate says supported
]


def fetch() -> Path:
    """Download the Track 3 stability table if it is not here yet."""
    if not RAW.exists():
        print(f"Fetching {SOURCE} -> {RAW.name}")
        urllib.request.urlretrieve(SOURCE, RAW)
    return RAW


def build() -> list[dict]:
    d = pd.read_csv(fetch())
    rates = d.groupby("allele")["thalf_hours"].apply(lambda s: (s > THRESHOLD_H).mean())
    claims = []
    for allele, answer, difficulty in PICKS:
        pool = d[d.allele == allele]
        pool = (pool[pool.thalf_hours >= THRESHOLD_H * MARGIN] if answer
                else pool[pool.thalf_hours <= THRESHOLD_H / MARGIN])
        if pool.empty:
            raise SystemExit(f"{allele}: no peptide clears the {MARGIN:g}x margin for "
                             f"answer={answer}. Pick a different allele.")
        row = pool.sort_values(["peptide"]).iloc[0]
        assert (row.thalf_hours > THRESHOLD_H) == answer, f"{allele}: margin check failed"
        base_rate = float(rates[allele])
        claims.append({
            "id": f"hla-{row.peptide.lower()}-{allele.split('*')[1].replace(':', '').replace('(', '-').replace(')', '')}",
            "allele": allele,
            "peptide": row.peptide,
            "claim": f"For peptide {row.peptide} presented by {allele}, "
                     f"the measured binding half-life is above {THRESHOLD_H:g} hour.",
            "criteria": f"Judged against the experimentally measured half-life for this exact "
                        f"peptide and allele, withheld from the solver. Supported if above "
                        f"{THRESHOLD_H:g} h, refuted if below.",
            "answer": "supported" if answer else "refuted",
            "measured_thalf_hours": float(row.thalf_hours),
            "allele_frac_above": round(base_rate, 3),
            "base_rate_answer": "supported" if base_rate > 0.5 else "refuted",
            "difficulty": difficulty,
            "prize": 100.0,
        })
    return claims


def main() -> None:
    claims = build()
    out = HERE / "claims.json"
    out.write_text(json.dumps(claims, indent=2) + "\n")

    hdr = f"{'peptide':10} {'allele':18} {'measured':>9}  {'truth':9} {'base rate':>9} {'guess wins?':11} {'level'}"
    print(hdr)
    print("-" * len(hdr))
    for c in claims:
        guess_right = c["base_rate_answer"] == c["answer"]
        print(f"{c['peptide']:10} {c['allele']:18} {c['measured_thalf_hours']:7.1f} h  "
              f"{c['answer']:9} {c['allele_frac_above']:8.0%} {'yes' if guess_right else 'NO':11} {c['difficulty']}")
    n_true = sum(c["answer"] == "supported" for c in claims)
    n_guess = sum(c["base_rate_answer"] == c["answer"] for c in claims)
    print(f"\n{n_true} true, {len(claims) - n_true} false. "
          f"Base-rate guessing scores {n_guess}/{len(claims)}.")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
