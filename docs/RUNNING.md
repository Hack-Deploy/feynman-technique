# Running Discovery Market

Everything needed to run, re-record and publish the app. The project overview is in the
[README](../README.md).

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone --recurse-submodules https://github.com/Hack-Deploy/feynman-technique.git
cd feynman-technique
uv sync
uv run python app.py
```

Open <http://localhost:8000>. No API key is needed to browse the pages or replay the recorded
runs.

`uv sync` installs the DiscoverPhysics packages from the pinned submodule as editable
dependencies. Never edit files under `vendor/`; wrap vendor behavior in project code.

## Pages

- **Pitch (`/`)** explains the market and the problem it addresses.
- **Live market (`/live`)** replays recorded real-model runs round by round: experiments,
  estimates, reasoning and the judge's ruling. Runs that released a probe at rest near a single
  source also get **Force versus distance**: pull readings derived from the recorded positions,
  the same launches replayed noise-free as the reference, and the claimed law where the claim
  names one (`poc/recorded_replay.py`, `law`). An explicitly enabled paid mode starts new runs.
- **How it works (`/simulation`)** walks through ForceBench and replays 88 published attempts
  by eight frontier models. Data from
  [ARA Labs (AgentNativeResearchLab)](https://huggingface.co/AgentNativeResearchLab), CC BY 4.0.
- **Marketplace (`/market`)** lists venues, bounties and agent submissions (not linked in the nav).

## Recorded runs

`/live` reads `attempts/fixtures/live/runs.jsonl`: 14 real-model runs recorded on
4 October 2026 with `poc.rerun_all` under the current claim protocol, up to 7 rounds per run,
$5.78 in API cost. Earlier recorded runs are archived under `attempts/archive/` (git-ignored).

The older Hugging Face snapshot can still be pulled, but it **replaces** the committed runs
(the current ones are archived under `attempts/archive/` first):

```bash
uv run python -m poc.hf_data pull      # downloads arushisinha98/discovery-market-live
```

## Run real models

The recorded and scripted modes make no API calls. To run real models, see
[the live-market instructions in `poc/README.md`](../poc/README.md#run-the-live-market-with-your-key):
the no-key spend preflight, the multi-model grid, the configured $50 hard cap, and the
append-only spend ledger.

Rounds per run are capped by `live.max_rounds` in `poc/live_models.yaml` (7). To re-record every
claim against every model:

```bash
uv run python -m poc.rerun_all --preflight                     # order and worst-case spend, no calls
ENABLE_LIVE=1 DM_MAX_USD=30 uv run python -m poc.rerun_all --purge --yes
```

Each claim gets its own seeded model order and stops after its first successful solve. If a run
is interrupted (for example a dropped connection), rerun without `--purge`: saved runs are kept
and open claims resume at their next untried model. The purge archives the live cache, summary,
benchmark store and transcripts; it never deletes them or touches the spend ledger. Set
`ANTHROPIC_API_KEY` in `poc/.env`.

The public record is opt-in: a model can pay 30 credits to read the failed runs for its claim.

## Publish

The hosted copy at <https://discovery-market.vercel.app> is static and read-only: every page plus
the JSON each read-only API endpoint returns. Paid runs are disabled there.

```bash
uv run python scripts/export_static.py                    # writes output/discovery-market/
cd output/discovery-market && vercel deploy --prod
```

To share run data on Hugging Face, set `HF_TOKEN` in `poc/.env` with write access:

```bash
uv run python -m poc.hf_data push --dry-run
uv run python -m poc.hf_data push
```

Push creates a private dataset unless `--public` is supplied, uploads only allow-listed live
records, and replaces the dataset's live data (older versions stay in its commit history).

## The simulated market

The original simulated market runs Tracks A and C from published pass rates
(`uv run python run.py`). A researcher funds a prize per world into escrow; agents bid when
`belief × prize > expected cost`; experiments are charged up front; and clear passes pay the
prize. Failed attempts enter the ledger when a world closes, unsolved prizes are refunded, and
credit conservation is checked throughout. The probabilities are stand-ins derived from
published benchmark scores, not measured market outcomes.

| | Track A | Track C |
|---|---|---|
| Question | Do stronger agents profit while weaker ones stop bidding? (H1); what prize clears each world? (H2) | At equal accuracy, does needing fewer experiments earn more? (H3) |
| Agents | opus-4.7, gpt-5.5, sonnet-4.6, qwen3.5-397b | mda, llm_opus, llm_opus_unthrottled |
| Worlds | all 11 | 6 (from Table 3) |
| Cost | 4–16 rounds × 1 credit | 8 or 41 experiments × price (0.25–2) |
| Sweep | prize 5–200 × odds source × 5 seeds | prize 20–100 × price × 5 seeds |

## Layout

| Path | What |
|---|---|
| `app.py`, `web/` | Local app and its frontend assets |
| `live_market.py`, `poc/` | Live discovery market: protocol, agents, spend controls, recorded replays |
| `dm/` | Attempt types, oracle, venues, replay, and market analysis |
| `scripts/export_static.py` | Static export for the hosted copy |
| `runner.py`, `market.py`, `analysis.py`, `run.py` | Simulated Track A/C market, sweeps, and analysis |
| `forcebench_demo.py` | One offline ForceBench attempt, run from the CLI |
| `vendor/discovery-agents/` | Pinned DiscoverPhysics submodule; never edit |
| `attempts/fixtures/` | Committed fixture attempt pools and recorded runs |
| `data/` | Published input tables and derived success probabilities |
| `tests/` | Engine, oracle, venue, live-market, and app tests |
| `PLAN.md`, `STATUS.md` | Project plan and progress log |
| `PITCH.md`, `DEMO.md` | Pitch script and presenter notes |

## Contributing

Work on a branch and open a pull request. Run `uv run pytest` before pushing. Add dependencies
with `uv add <pkg>` (or `uv add --dev <pkg>`) and commit `uv.lock`.
