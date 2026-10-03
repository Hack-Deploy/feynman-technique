# Bounty benchmark on DiscoverPhysics (proof of concept)

The market from `slides.html`, run on the 11 DiscoverPhysics worlds. A researcher posts a
hypothesis with resolution criteria and a prize. The AI scientist plans experiments, states its
chance of a clear answer, bids or walks away, pays the lab, and gets the prize only for a clear,
correct verdict. Failed runs go into a public record that later runs see.

Everything you'd want to change is in [`config.yaml`](config.yaml): hypotheses, resolution
criteria, prizes, the round fee, per-part experiment costs, an optional budget, and the round cap.

## What the agent does each round

Every reply has `<assessment>` and `<p_success>` (chance of ending with a clear, correct verdict),
plus one action:

- `<run_experiment>`: priced from its parts (measurements, duration, particles, custom properties)
  and charged if it runs;
- `<run_mse_fit>`: test a candidate law against its own data;
- `<verdict>supported | refuted | inconclusive</verdict>` with `<evidence>`;
- `<withdraw>reason</withdraw>`: free in the first reply (walking away), otherwise it pays that round.

Every round costs `round_fee`. The answer key (`answer` in `config.yaml`) is never shown.

## Memory

Each run starts fresh. The only carry-over is the public record of earlier failed runs on the
same hypothesis: model, outcome, rounds, spend, stated p, withdrawal reason, and the raw
experiment data. Their verdicts, evidence and assessments are hidden, because a wrong verdict
would give the answer away.

## API key

Copy `poc/.env.example` to `poc/.env` (git-ignored; an empty one is already there) and fill in
`ANTHROPIC_API_KEY`. `poc.bench` loads it; values set in your shell win.

## Commands

```bash
uv run pytest tests/test_poc.py -q
uv run python -m poc.bench --fake                       # scripted LLM, real simulator, no API
ENABLE_LIVE=1 DM_MAX_USD=20 uv run python -m poc.bench --models claude-sonnet-4-6 \
    --seeds 0 1 2 --usd-per-call 0.05                   # live; prints projected spend first
uv run python -m poc.report --dashboard                 # summary + output/poc_dashboard.html
```

The report covers, per model: outcomes, correct verdicts, Brier score for the final p and for
the bid-time p, a reliability table, experiments, spend, cost per correct
verdict, and profit. It also breaks results down by how many failed runs a run had seen.

## Files

`config.yaml`/`config.py` settings · `pricing.py` prices and account · `protocol.py` prompt text,
reply parsing, public record · `agent.py` the vendor agent loop with bounty terms ·
`attempt.py` one run · `bench.py` all runs and resolution · `report.py` · `fake_llm.py` ·
`dashboard.html` runs page (opens with Dummy A / Dummy B placeholder runs; `report.py --dashboard`
writes a copy with real runs).
The vendor code is not edited.
