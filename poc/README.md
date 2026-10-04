# Bounty benchmark on DiscoverPhysics (proof of concept)

The market from `slides.html`, run on the 11 DiscoverPhysics worlds. A researcher posts a
hypothesis with resolution criteria and a prize. The AI scientist plans experiments, states its
chance of a clear answer, bids or walks away, pays the lab, and gets the prize only for a clear,
correct verdict. Failed runs go into an opt-in public record that a later model can buy for
30 credits. A successful solve closes that claim to later real runs.

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

Each run starts fresh. The public record contains only failed runs on the same hypothesis:
model, outcome, rounds, spend, stated p, withdrawal reason, and raw experiment data. Their
verdicts, evidence and assessments stay hidden. Models can buy that record for 30 credits,
paid to the market and never refunded; the purchase is optional, happens once, and uses no
round. If no failed runs exist, there is nothing to buy.

## Commands

```bash
uv run pytest tests/test_poc.py -q
uv run python -m poc.bench --fake                       # scripted LLM, real simulator, no API
uv run python -m poc.demo_grid --fake                   # multi-model recorded demo, no API
uv run python -m poc.demo_grid --preflight              # worst-case live spend, no key needed
uv run python -m poc.rerun_all --preflight               # claim-major randomized order and spend
uv run python -m poc.report --dashboard                 # summary + output/poc_dashboard.html
```

`--fake` writes the deterministic recorded demo to
`attempts/fixtures/live/scripted_demo.jsonl` and its derived
`scripted_demo.summary.json`; it makes no API calls.

## Run the live market with your key

Live-market observations include independent Gaussian noise: positions σ = 0.075 and velocities
σ = 0.05. The Poc wrapper adds velocity noise because the vendor executor only noises positions.
Results can be inconclusive; repeating an experiment in the same or a later round gives a fresh
reading and is charged at full price.

```bash
cp poc/.env.example poc/.env
# Set ANTHROPIC_API_KEY in poc/.env.
uv run python -m poc.demo_grid --preflight
ENABLE_LIVE=1 DM_MAX_USD=50 uv run python -m poc.demo_grid
# Add --yes to skip the confirmation prompt.
git add attempts/fixtures/live/runs.jsonl attempts/fixtures/live/runs.summary.json && git commit -m "Live demo runs"
uv run python app.py
# Open http://127.0.0.1:8000/live and scroll to "3 · Recorded runs".
```

For a fresh run of every configured claim against every configured model, first inspect the
claim-major order and worst-case spend, then confirm the archive and rerun. Each claim gets an
independent seeded model permutation in config order. Later models on a claim are skipped after
the first success; unsolved claims resume with the next untried model in their stored order:

```bash
uv run python -m poc.rerun_all --preflight
ENABLE_LIVE=1 uv run python -m poc.rerun_all --purge
```

Set `DM_MAX_USD=50` in `poc/.env` (or the shell). The purge archives the current live cache,
summary, benchmark store and transcript directory under `attempts/archive/`; it does not touch
the spend ledger or scripted-demo fixtures. The command asks once before archiving and running.

An interrupted call (crash or Ctrl-C mid-request) leaves its reservation open and counted at its
worst case, so the cap stays safe; Anthropic may still have billed tokens for that request.
Thinking models use `max_tokens: 16000` for both reasoning and the answer. Anthropic's explicit
client timeout is 600 seconds; replies stopped at the token limit are flagged in the round and
replay views.

The spend ledger at `attempts/live_spend.jsonl` is git-ignored, shared by the app and CLI, and
cumulative. Open reservations count toward the cap until settled or voided. The effective cap
is the lower of `DM_MAX_USD` and `live.max_usd` in `poc/live_models.yaml` (both default to $50);
the configured hard cap is reported when a larger environment value is supplied.

The report covers, per model: outcomes, correct verdicts, Brier score for the final p and for the
bid-time p, a reliability table, experiments, spend, cost per correct verdict, and profit. It also
breaks results down by how many failed runs a run had seen.

## Share / warm start

The default dataset is public, so pulling it needs no token. `--repo ORG/NAME` overrides the
default; `DM_HF_REPO` is an optional environment override. Set `HF_TOKEN` in `poc/.env` with
write access to push:

```bash
uv run python -m poc.hf_data push --dry-run
uv run python -m poc.hf_data push
uv run python -m poc.hf_data pull
```

Push is private unless `--public` is supplied. The dry run makes no network calls.
Push replaces the dataset's allow-listed live data with yours; older versions stay in the
dataset's commit history.
Pull replaces the local allow-listed live record with the dataset snapshot, archiving
the old files and directories under `attempts/archive/`.

## Files

`config.yaml`/`config.py` settings · `pricing.py` prices and account · `protocol.py` prompt text,
reply parsing, public record · `agent.py` the vendor agent loop with bounty terms ·
`attempt.py` one run · `bench.py` all runs and resolution · `report.py` · `fake_llm.py` ·
`dashboard.html` runs page (opens with Dummy A / Dummy B placeholder runs; `report.py --dashboard`
writes a copy with real runs).
The vendor code is not edited.
