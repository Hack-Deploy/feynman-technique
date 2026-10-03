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

## Commands

```bash
uv run pytest tests/test_poc.py -q
uv run python -m poc.bench --fake                       # scripted LLM, real simulator, no API
uv run python -m poc.demo_grid --fake                   # multi-model recorded demo, no API
uv run python -m poc.demo_grid --preflight              # worst-case live spend, no key needed
uv run python -m poc.report --dashboard                 # summary + output/poc_dashboard.html
```

`--fake` writes the deterministic recorded demo to
`attempts/fixtures/live/scripted_demo.jsonl` and its derived
`scripted_demo.summary.json`; it makes no API calls.

## Run the live market with your key

```bash
cp poc/.env.example poc/.env
# Set ANTHROPIC_API_KEY in poc/.env.
uv run python -m poc.demo_grid --preflight
ENABLE_LIVE=1 DM_MAX_USD=5 uv run python -m poc.demo_grid
# Add --yes to skip the confirmation prompt.
git add attempts/fixtures/live/runs.jsonl attempts/fixtures/live/runs.summary.json && git commit -m "Live demo runs"
uv run python app.py
# Open /live and choose "Recorded runs".
```

The spend ledger at `attempts/live_spend.jsonl` is git-ignored, shared by the app and CLI, and
cumulative. Open reservations count toward the cap until settled or voided. The effective cap
is the lower of `DM_MAX_USD` and `live.max_usd` in `poc/live_models.yaml` (default $5); rerun the
same command to resume from the cached runs.

The report covers, per model: outcomes, correct verdicts, Brier score for the final p and for the
bid-time p, a reliability table, experiments, spend, cost per correct verdict, and profit. It also
breaks results down by how many failed runs a run had seen.

## Files

`config.yaml`/`config.py` settings · `pricing.py` prices and account · `protocol.py` prompt text,
reply parsing, public record · `agent.py` the vendor agent loop with bounty terms ·
`attempt.py` one run · `bench.py` all runs and resolution · `report.py` · `fake_llm.py` ·
`dashboard.html` runs page (opens with Dummy A / Dummy B placeholder runs; `report.py --dashboard`
writes a copy with real runs).
The vendor code is not edited.
