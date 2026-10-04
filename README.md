# Discovery Market

A market where AI scientists get paid only for clear results. A researcher posts
a hypothesis and a prize; experiment designers bid when they expect a result to
be worth the cost; labs are paid for experiments; and every attempt, including
failures, is recorded. The judge pays only for a clear answer that passes its
precommitted tests.

## The app

The local app has three pages:

- **Vision (`/`)** explains the market and the problem it addresses.
- **How it works (`/simulation`)** walks through ForceBench for a general audience.
  Its sixth section replays 88 published attempts by eight frontier models:
  who solved what, the clearing prize per world, profit per model, and whether
  the models knew their chances. The data are from
  [ARA Labs (AgentNativeResearchLab)](https://huggingface.co/AgentNativeResearchLab), CC BY 4.0.
- **Live market (`/live`)** has a free scripted mode, an explicitly enabled real
  mode, and a recorded-runs comparison. The committed recorded examples are
  scripted stand-ins; no real paid runs have been recorded.

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.12.

```bash
git clone --recurse-submodules https://github.com/Hack-Deploy/feynman-technique.git
cd feynman-technique
uv sync
uv run python app.py
```

Open <http://localhost:8000>. No API key is needed to browse the pages or use the
scripted and recorded examples.

`uv sync` installs the DiscoverPhysics packages from the pinned submodule as editable
dependencies. Never edit files under `vendor/`; wrap vendor behavior in project code.

## Quick start with recorded data

```bash
git clone --recurse-submodules https://github.com/Hack-Deploy/feynman-technique.git
cd feynman-technique
uv sync
uv run python -m poc.hf_data pull      # downloads arushisinha98/discovery-market-live
uv run python app.py                   # open /live → Recorded runs
```

No Anthropic key is needed to view the recorded data. The default dataset is public, so no
token is needed; `HF_TOKEN` is only needed for a private fork or copy, set in `poc/.env`
with read access. Pull archives replaced local files under `attempts/archive/` and fills
`attempts/fixtures/live/runs.jsonl`, which `/live` uses for Recorded runs. `poc.rerun_all`
reads its cached run keys and `order_seed`, so completed runs are skipped and the plan resumes.

## Run the live market with your key

The `/live` page's scripted and recorded modes make no API calls. To run real models,
follow [the live-market instructions in `poc/README.md`](poc/README.md#run-the-live-market-with-your-key):
they cover the no-key spend preflight, the multi-model grid, the configured $50 hard cap,
committing recorded runs, and the persistent append-only spend ledger.

To preview a fresh randomized model order for every claim and archive the current live
record before rerunning the full grid:

```bash
uv run python -m poc.rerun_all --preflight
ENABLE_LIVE=1 uv run python -m poc.rerun_all --purge
```

The purge archives the live cache, summary, benchmark store, and transcripts; it never
deletes them or touches the spend ledger. The run still requires a positive `DM_MAX_USD`
cap (for example, set `DM_MAX_USD=50` in `poc/.env`) and an API key.

## Share / warm start

To push data, set `HF_TOKEN` in `poc/.env` with write access. `--repo ORG/NAME` overrides
the default dataset; `DM_HF_REPO` is an optional environment override.

```bash
uv run python -m poc.hf_data push --dry-run
uv run python -m poc.hf_data push
```

Push creates a private dataset unless `--public` is supplied and uploads only allow-listed
live records. `--dry-run` lists files and sizes without network calls.

## How the simulation works

The original simulated market runs Tracks A and C. A researcher funds a prize per
world into escrow; agents bid when `belief × prize > expected cost`; experiments are
charged up front; and clear passes pay the prize. Failed attempts enter the ledger
when a world closes. Unsolved prizes are refunded, and credit conservation is checked
throughout. These probabilities are stand-ins derived from published benchmark
scores, not measured market outcomes.

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
| `app.py`, `web/` | Three-page local app and its frontend assets |
| `forcebench_demo.py` | One offline ForceBench attempt, run from the CLI |
| `runner.py`, `market.py`, `analysis.py`, `run.py` | Simulated Track A/C market, sweeps, and analysis |
| `report.py`, `report_template.html` | Legacy CLI export; not served by the app |
| `slides.html` | Pitch deck; not part of the app |
| `docs/` | Technical notes, review findings, and judge evaluation |
| `poc/` | Live discovery-market implementation, spend controls, and instructions |
| `dm/` | Attempt types, oracle, venues, replay, and market analysis |
| `vendor/discovery-agents/` | Pinned DiscoverPhysics submodule; never edit |
| `attempts/fixtures/` | Committed fixture attempt pools and recorded demo data |
| `data/` | Published input tables and derived success probabilities |
| `config.yaml` | Simulation and market parameters |
| `tests/` | Engine, oracle, venue, live-market, and app tests |
| `PLAN.md`, `STATUS.md` | Project plan and progress log |
| [DEMO.md](DEMO.md) | Presenter script for the three pages |

## Contributing

Work on a branch and open a pull request. Run `uv run pytest` before pushing.
Add dependencies with `uv add <pkg>` (or `uv add --dev <pkg>`) and commit `uv.lock`.
