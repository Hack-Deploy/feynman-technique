# Discovery Market

A market where AI scientists get paid only for clear results. A researcher posts
a hypothesis and a prize; experiment designers bid only when they're confident
enough to cover the cost; labs charge upfront to run the experiment; the prize
pays out only for a clear answer. Every run is logged, including failures.

This repo is the proof of concept: the full market simulated on
[DiscoverPhysics](https://arxiv.org/abs/2605.26087) — 11 physics worlds where the
true answer is known, so every payout can be checked.

## Slide deck

- **[slides.pdf](slides.pdf)**: view the deck right here on GitHub.
- **[slides.html](slides.html)**: the presentable version. Download it and open it in a browser, then use ← → to move through the slides. It can also be served with GitHub Pages.
- **[novelty-review.md](novelty-review.md)**: the literature and startup review behind slide 2, with sources.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                      # install Python 3.12 + dependencies into .venv
uv run python run.py         # run all sweeps → output/{events,ledger,summary}.json
uv run pytest                # run the test suite
```

A full run takes about a second. `output/` is not committed; regenerate it.

## How the simulation works

Each run is 200 ticks. At tick 0 the researcher funds a prize per world into
escrow and each agent gets 100 credits. Every tick, each agent bids on each open
world where `belief × prize > expected cost`. Bids are shuffled, charged upfront
to the lab, and resolved with the agent's true pass probability. A pass pays the
prize and closes the world; a failure is logged to the ledger, hidden until the
world closes. Unsolved prizes are refunded at the end. Credits are conserved and
checked every tick.

| | Track A | Track C |
|---|---|---|
| Question | Do stronger agents profit while weaker ones stop bidding? (H1) What prize clears each world? (H2) | At equal accuracy, does needing fewer experiments earn more? (H3) |
| Agents | opus-4.7, gpt-5.5, sonnet-4.6, qwen3.5-397b | mda, llm_opus, llm_opus_unthrottled |
| Worlds | all 11 | 6 (from Table 3) |
| Cost | 4–16 rounds × 1 credit | 8 or 41 experiments × price (0.25–2) |
| Sweep | prize 5–200 × odds source (raw, calibrated, power) × 5 seeds | prize 20–100 × price × 5 seeds |

All probabilities are stand-ins derived from published benchmark scores, not
measured market outcomes.

## Layout

| Path | What |
|---|---|
| `market.py` | Market engine: pure logic, pluggable cost models, emits events + ledger rows |
| `runner.py` | Builds Track A / Track C runs and sweeps, writes `output/` |
| `analysis.py` | Recomputes everything from the event log; H1–H3 verdicts → `summary.json` |
| `run.py` | Entry point: sweeps → save → analyse → print verdicts |
| `data/table{1,2,3}_*.csv` | Source tables: explanation scores, model pass@1, MDA pass rates |
| `data/success_table.csv` | Per-world pass probabilities (`p = score^γ`, γ fit to pass@1); built by `data/build_success_table.py`. Used as the `power` odds source in Track A. |
| `config.yaml` | Intended run parameters (not yet read by the runner) |
| `tests/` | Engine and analysis tests |
| `STATUS.md` | Progress log |

## Contributing

Work on a branch and open a pull request against `main`. Run `uv run pytest`
before pushing. Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`),
and commit `uv.lock`.
