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
- **[novelty-review.md](novelty-review.md)**: literature and startup review with sources, backing the claims on slide 1.

## Quick start

Requires [uv](https://docs.astral.sh/uv/).

```bash
git submodule update --init  # fetch DiscoverPhysics into vendor/discovery-agents (pinned)
uv sync                      # install Python 3.12 + dependencies into .venv
uv run python run.py         # run all sweeps → output/{events,ledger,summary}.json
uv run pytest                # run the test suite
uv run python app.py         # results app at http://localhost:8000
uv run python report.py      # same dashboard as one file: output/report.html
```

`uv sync` installs the two DiscoverPhysics packages (`PhysicsSchool`, `ScienceAgent`) from the
submodule as editable path dependencies (see `[tool.uv.sources]` in `pyproject.toml`), plus
`requests`, which `scienceagent` imports but does not declare. Never edit files under
`vendor/`; wrap them in `dm/` instead.

`app.py` opens a page with buttons for the simulation and the tests, and shows
the results: the prize each world needs, agent balances payment by payment,
mean profit per prize, the public ledger and the verdicts. It uses only the
standard library and listens on localhost. `report.py` writes the same page
with the data baked in, for sharing.

**Live bounties.** The app's "Post a bounty" form sends a biology hypothesis,
success criterion and prize to Claude (`claude-opus-5-5`), which returns a
protocol priced from a fixed lab price list, a stated probability of a clear
answer and a biosafety level. `bounty.py` then applies the market rule in code:
bid only if p × prize > cost, never above BSL-2. For live mode, put your key in
a `.env` file in the repo root (git ignores it) and restart the app:

```
ANTHROPIC_API_KEY=sk-ant-...
```

Without a key the form shows a saved, clearly labelled example.

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
| `app.py` | Local results app: bounty form, run buttons, dashboard |
| `bounty.py` | Bounty → Claude experiment design → bid decision (rule applied in code) |
| `report.py`, `report_template.html` | Dashboard data and page; `report.py` also exports `output/report.html` |
| `data/table{1,2,3}_*.csv` | Source tables: explanation scores, model pass@1, MDA pass rates |
| `data/success_table.csv` | Per-world pass probabilities (`p = score^γ`, γ fit to pass@1); built by `data/build_success_table.py`. Used as the `power` odds source in Track A. |
| `config.yaml` | Intended run parameters (not yet read by the runner) |
| `tests/` | Engine and analysis tests |
| `STATUS.md` | Progress log |

## Contributing

Work on a branch and open a pull request against `main`. Run `uv run pytest`
before pushing. Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`),
and commit `uv.lock`.
