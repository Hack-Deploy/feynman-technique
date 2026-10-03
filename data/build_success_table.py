"""Build success_table.csv: per-model, per-world pass probabilities for the market.

Source: DiscoverPhysics (Wiemann, Smith et al., arXiv:2605.26087v2).
  - Per-world mean explanation scores: Figure 5 heatmap (mean over 5 seeds).
  - Per-model pass@1: Table 1.

Why not use the explanation score directly as the pass probability?
A pass needs explanation score >= 0.9 AND normalized MSE < 0.1, so mean
explanation scores overstate success. Opus 4.7 averages 0.61 but passes 26.4%.

Mapping: p = s ** gamma, with one gamma per model, chosen so the model's mean
p over all 22 worlds equals its published pass@1. The power curve keeps
0 -> 0 and 1 -> 1 (a world scored 1.00 on every seed stays a near-sure pass)
and pushes middling scores down, which is how a >= 0.9 threshold behaves.

Run: python3 data/build_success_table.py   (stdlib only)
"""

import csv
from pathlib import Path

# Figure 5 column order (sorted by Opus 4.7 difficulty). "Private N" worlds are
# redacted in the paper; they count toward calibration but aren't in the market.
WORLDS = [
    "Private 1", "Gravity", "Private 2", "Yukawa", "Hubble", "Private 3",
    "Ether", "Oscillator", "Private 4", "Coulomb", "Private 5", "Circle",
    "Extra dims", "Private 6", "Fractional", "Private 7", "Private 8",
    "Dark matter", "Three species", "Private 9", "Private 10", "Private 11",
]

# Figure 5 rows, transcribed from the PDF text layer.
HEATMAP = {
    "Claude Opus 4.7": [1.00, 0.94, 0.80, 0.96, 0.90, 0.54, 0.76, 0.72, 0.62, 0.70, 0.72, 0.48, 0.54, 0.54, 0.48, 0.48, 0.54, 0.38, 0.40, 0.30, 0.28, 0.44],
    "Claude Sonnet 4.6": [0.40, 0.50, 0.26, 0.28, 0.24, 0.10, 0.38, 0.38, 0.32, 0.24, 0.14, 0.36, 0.16, 0.18, 0.34, 0.22, 0.26, 0.22, 0.34, 0.24, 0.28, 0.20],
    "Claude Haiku 4.5": [0.56, 0.36, 0.14, 0.34, 0.12, 0.08, 0.22, 0.20, 0.14, 0.62, 0.16, 0.20, 0.16, 0.26, 0.46, 0.16, 0.24, 0.16, 0.16, 0.10, 0.20, 0.06],
    "GPT-5.5": [0.96, 0.96, 0.56, 0.90, 0.64, 0.56, 0.76, 0.90, 0.74, 0.74, 0.66, 0.54, 0.36, 0.72, 0.48, 0.46, 0.36, 0.22, 0.40, 0.46, 0.12, 0.44],
    "GPT-5.4": [0.60, 0.68, 0.14, 0.42, 0.22, 0.18, 0.24, 0.10, 0.24, 0.52, 0.30, 0.30, 0.20, 0.36, 0.48, 0.14, 0.42, 0.14, 0.38, 0.16, 0.06, 0.00],
    "GPT-oss-120b": [0.40, 0.62, 0.04, 0.36, 0.18, 0.12, 0.20, 0.26, 0.20, 0.56, 0.16, 0.12, 0.18, 0.12, 0.32, 0.06, 0.22, 0.26, 0.14, 0.12, 0.06, 0.04],
    "GPT-oss-20b": [0.34, 0.42, 0.02, 0.04, 0.12, 0.08, 0.17, 0.06, 0.12, 0.30, 0.08, 0.08, 0.08, 0.24, 0.38, 0.08, 0.16, 0.14, 0.10, 0.06, 0.00, 0.00],
    "Qwen3.5-397B": [0.70, 0.76, 0.24, 0.40, 0.20, 0.10, 0.36, 0.26, 0.40, 0.30, 0.36, 0.46, 0.24, 0.46, 0.46, 0.16, 0.42, 0.30, 0.26, 0.30, 0.10, 0.10],
    "Qwen3.2-Instruct": [0.54, 0.50, 0.22, 0.46, 0.12, 0.16, 0.12, 0.16, 0.26, 0.30, 0.14, 0.34, 0.20, 0.30, 0.34, 0.04, 0.18, 0.18, 0.08, 0.12, 0.08, 0.14],
    "Llama-3.3-70B": [0.28, 0.14, 0.06, 0.14, 0.08, 0.02, 0.06, 0.08, 0.12, 0.16, 0.04, 0.12, 0.00, 0.06, 0.08, 0.12, 0.12, 0.10, 0.07, 0.10, 0.12, 0.02],
    "Kimi-K2.5": [0.76, 0.76, 0.15, 0.40, 0.32, 0.12, 0.24, 0.16, 0.17, 0.36, 0.40, 0.34, 0.22, 0.30, 0.34, 0.20, 0.22, 0.28, 0.24, 0.16, 0.04, 0.05],
}

# Table 1 pass@1 (%), mean over all 22 worlds.
PASS_AT_1 = {"Claude Opus 4.7": 26.4, "GPT-5.5": 21.7, "Claude Sonnet 4.6": 1.8, "Qwen3.5-397B": 4.5}

MARKET_MODELS = ["Claude Opus 4.7", "GPT-5.5", "Claude Sonnet 4.6", "Qwen3.5-397B"]
PUBLIC_WORLDS = [w for w in WORLDS if not w.startswith("Private")]


def fit_gamma(scores, target):
    """Find gamma with mean(s ** gamma) == target. Mean decreases in gamma, so bisect."""
    lo, hi = 1e-3, 200.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if sum(s ** mid for s in scores) / len(scores) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    assert all(len(row) == len(WORLDS) for row in HEATMAP.values())
    out = Path(__file__).with_name("success_table.csv")
    rows = []
    for model in MARKET_MODELS:
        scores = HEATMAP[model]
        gamma = fit_gamma(scores, PASS_AT_1[model] / 100)
        for world, s in zip(WORLDS, scores):
            if world in PUBLIC_WORLDS:
                rows.append({
                    "model": model,
                    "world": world,
                    "explanation_score": f"{s:.2f}",
                    "pass_prob": f"{s ** gamma:.4f}",
                    "gamma": f"{gamma:.3f}",
                })
        mean_p = sum(s ** gamma for s in scores) / len(scores)
        print(f"{model:18s} gamma={gamma:6.3f}  mean expl={sum(scores)/len(scores):.3f}  "
              f"mean p (22 worlds)={mean_p:.3f}  target={PASS_AT_1[model]/100:.3f}")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
