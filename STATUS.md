# Discovery Market – Status Log

## P0 Setup
- **Built**: Repository layout, config.yaml, data/*.csv from Tables 1-3.
- **Check**: `test_table1_*`, `test_table2_*`, `test_table3_*`, `test_config_loads` — all 7 PASSED.
  CSVs match the specification tables exactly.
- **Result**: ✅ PASS

## P1 Engine (market.py)
- **Built**: Pure-logic market engine with pluggable cost models (Track A / Track C),
  Beta-distribution beliefs, credit conservation checks, event and ledger generation.
- **Checks (33 tests)**:
  - Credit conservation over 5 seeds (simple + full Track A + Track C): 15 PASSED
  - Bid-rule boundary (no bid at/below, bids above): 4 PASSED
  - Cancelled bids cost nothing: 1 PASSED
  - Identical events for same seed (determinism): 2 PASSED
  - Ledger rows hidden until disclosure: 3 PASSED
  - Refunds at end of run: 2 PASSED
  - Track C cost model: 2 PASSED
  - Probability sources (raw, calibrated): 3 PASSED
  - Calibrated odds at prize 20 expect 0 bids: 1 PASSED
- **Result**: ✅ PASS — all 40 tests passed in 1.08s.

## P2/P3 Sweeps (runner.py)
- **Built**: Track A (2 odds sources × 5 prizes × 5 seeds = 50 runs) and
  Track C (4 prices × 3 prizes × 5 seeds = 60 runs). 8,545 events, 1,413 ledger rows.
- **Result**: ✅ PASS

## P4 Analysis (analysis.py, run.py)
- **Built**: Every number recomputed from the event log; H1/H2/H3 verdicts in
  `output/summary.json`.
- **Result**: ✅ PASS

## Fixes (2026-10-03)
- **Profit double-count**: profits were 100 credits too low (starting credits
  subtracted twice). Opening balances are now `account_funded` events at tick 0.
- **Nondeterminism**: runs varied with `PYTHONHASHSEED` (worlds iterated from a
  set). Now iterated in config order; cross-process determinism test added.
- **H1 verdict**: agents ranked by pass@1; each prize classified SUPPORTED /
  PARTIAL / NOT SUPPORTED. Weaker agents never bid at any prize (priced out by
  their prior), so "stop bidding" is not yet tested dynamically.
- **Checks**: 57 tests (42 engine, 15 analysis) — all PASSED.

## Stage 0 Cleanup (2026-10-03)
- `.gitignore` conflict markers resolved.
- Merged `data/success-table`: `data/success_table.csv` regenerates identically
  from `data/build_success_table.py`; each model's mean p matches its pass@1.
- README written; this log brought up to date.

## Stage 1: power odds (2026-10-03)
- **Built**: `success_table.csv` wired in as a third Track A odds source,
  `power` (p = score^γ, γ fit per model so mean p over 22 worlds = pass@1).
  `load_success_table()` maps its display names onto Table 1 names.
- **Why**: `calibrated` scales scores linearly, which caps strong worlds far
  too low (opus-4.7 gravity 0.41 vs 0.79) and props up weak ones.
- **Result**: under `power`, the hidden-structure worlds need the top prize
  (circle, extra_dimensions, dark_matter clear only at 200) and three_species
  is never solved. H1 SUPPORTED at prize 100 and 200.
- **Checks**: 3 new tests (known values, full coverage, order preserved);
  60 tests PASSED.

## Results app (2026-10-03)
- **Built**: `app.py` (local app: run simulation, run tests, view results) and
  `report.py` + `report_template.html` (same dashboard, exportable as
  `output/report.html`). Charts: clearing prize per world, agent balances
  payment by payment, mean profit per prize, public ledger, verdicts.
- **Checks**: `tests/test_report.py` checks dashboard balances and ledger
  counts against `summary.json`; 62 tests PASSED.

## Ledger leak fix (2026-10-03)
- **Fixed**: ledger rows stored each solver's hidden true pass probability in
  `outcome.metric`. Rows are public once disclosed, so that leaked the answer
  the market is meant to discover. `outcome` now holds only `passed`.
- **Checks**: `test_ledger_hides_true_probability`; 63 tests PASSED.

## Live bounties (2026-10-03)
- **Built**: `bounty.py` + a "Post a bounty" form in the app. A biology
  hypothesis, criterion and prize go to Claude, which returns a protocol priced
  from a fixed price list, a stated probability and a biosafety level. The bid
  rule (p × prize > cost, BSL-2 max) runs in code. No key → labelled example.
- **Checks**: `tests/test_bounty.py` (cost from price list, bid threshold,
  BSL block, clamping, example fallback, empty form); 69 tests PASSED.

## Open issues
- `config.yaml` is not read by the runner/analysis; values are hard-coded.
- H3 holds by construction: `llm_opus_unthrottled` reuses mda's pass rates.
- Disclosed ledger rows don't update other agents' beliefs (Stage 3).

---

# Real attempts (branch `real-attempts`; plan in `PLAN.md`)

## Phase 0: hygiene and vendor (2026-10-03)
- **Built**:
  - `vendor/discovery-agents` submodule pinned at `450818fa81c61ed0030351ee7f634c9e09d412bc`;
    installed by `uv sync` as editable path deps (`physchool`, `scienceagent`) plus `requests`
    (undeclared vendor import) and `scipy`. New deps pulled in by the vendor: jax, anthropic,
    openai, pyyaml.
  - `.gitignore`: short version; ignores `attempts/*` except `attempts/fixtures/`, and
    `vendor/**/results/`.
  - Leak fixed: simulated ledger rows now store `outcome.metric = null` (was the true probability).
  - `CostModel.expected_cost()` on both cost models; `market._expected_cost` and its
    `isinstance` dispatch removed.
  - `tests/fixtures/baseline_sha256.json`: hashes of `run.py` outputs after this phase, the
    reference for Phase 1's byte-identity check.
- **Check**:
  - Old vs new code: `events.json` and `summary.json` byte-identical; `ledger.json` identical
    except `metric` → `null` (1,413 rows).
  - Smoke test builds `gravity` (nbody, σ = 0.075) and runs one experiment through the vendor
    executor.
  - Facts verified for later phases: the 1/r fixture (k = 1/2π) scores mean_pos_error 0.0000 on
    gravity and 0.9850 on yukawa (default cases); `np.var` of flattened noise-free test
    positions reproduces the vendor's `_WORLD_VARS` exactly for the six two-particle worlds.
  - 63 tests passed (57 existing + 6 new).
- **Result**: ✅ PASS
- **Conflicts with the task prompt**: recorded in `PLAN.md` §1 (C1–C14). The main ones: the
  normalising variances are in the vendor repo (`run_benchmark._WORLD_VARS`); ForceBench is
  only in MDA arXiv v1–v3 (cited as v3); ARA used relative noise σ = 0.075·√Var(world) while
  the vendor default is absolute 0.075; ARA `episode.json` does record experiment counts.

## Phase 1: engine generalisation (2026-10-03)
- **Built**:
  - `market.OutcomeSource` protocol: `available(agent, world)`, `draw(agent, world, rng) →
    (passed, cost_detail, attempt_id | None)`. `market.BernoulliTable` is today's coin flip
    (same rng calls in the same order: cost, then pass). `dm/outcomes.ReplayPool` replays
    `AttemptRecord`s per (solver, world), without replacement within a run, with replacement
    across runs; it is also the run's cost model (expected/max cost of what is left in the
    pool), so an empty pool means the solver cannot bid. Cost functions: `rounds_cost`,
    `experiments_cost`, `recorded_cost`.
  - Real-attempt mode (`MarketRun.state_confidence=True`): each bid emits `attempt_started`,
    `confidence_stated` (the record's stated p, else the agent's belief mean),
    the charge with `count`, `attempt_submitted`, `verdict_issued`; `insufficient_credits` once
    per (agent, world) when an agent wants to bid but cannot afford it. With
    `MarketRun.preregs`, `prize_posted` carries the commitment, `prereg_committed` follows it,
    and `prereg_revealed` (full preregistration) is emitted when the world closes or is refunded.
  - `Event` gains optional `count`, `p`, `commitment`, `attempt_id`, `detail`, serialised only
    when set. Ledger `outcome.metric` holds the measured normalised MSE for replays, else null.
  - `dm/types.py` (`Preregistration`, `AttemptRecord` + an `extra` dict for provenance,
    `InsufficientCredits`), `dm/store.py` (append-only JSONL; a later line with the same id
    supersedes), fixture pool `attempts/fixtures/replay_pool.jsonl` (8 hand-made records).
- **Check**: legacy runs through `BernoulliTable` reproduce the Phase 0 hashes of
  `events.json`, `ledger.json`, `summary.json` exactly (test). ReplayPool on the fixture pool:
  conservation over 5 seeds × 2 cost functions, determinism, no record drawn twice in a run,
  records reused across runs, no bids on an empty pool, charges = experiments × price, verdicts
  and stated p match the records, lifecycle order. 94 tests passed.
- **Result**: ✅ PASS

## Merge of `main` (85b6090: PRs #2 power odds, #3 ledger leak, #4 dashboard)
- Both this branch (Phase 0) and PR #3 fixed the ledger leak. Kept PR #3's convention:
  simulated ledger rows record only `{"passed": ...}` (no `metric` key). Replayed real attempts
  additionally record `metric` = their measured normalised MSE (not hidden information).
- The Phase 1 byte-identity baseline (`tests/fixtures/baseline_sha256.json`) is regenerated from
  **`main`'s own code** at 85b6090 (including the new `power` odds source), and this branch's
  engine reproduces those three files exactly.

## Phase 2: the oracle — IN PROGRESS (not yet checked)
- **Built so far**: `dm/oracle/` (`make_prereg`, `score`, `explain_score`), `dm/oracle/cases.py`
  (hidden cases + normalising variance), `dm/oracle/_worker.py` (scoring subprocess: timeout,
  network disabled, API keys stripped from env), `dm/settle.py` (only caller of the oracle),
  `dm.types.SubmittedAttempt`.
- **Verified**: the normalising variance convention reproduces all 11 vendor `_WORLD_VARS` on the
  default cases (two-particle: `pos2`; multi-particle: every agent-facing particle, even where only
  probes are scored). Hidden layouts use fixed seeds (42/123), so the oracle can rebuild worlds.
- **Hidden-case design**: compared three designs over test seeds 0–2 against true laws and wrong
  laws. Chose V1 (base launch r0 ∈ U[3, 6], tangential 0.2–0.5, all cases measured at t = 1..10):
  true laws score nMSE ≤ 5e-6 on gravity/yukawa/fractional/oscillator/coulomb_easy; the 1/r law
  fails yukawa on all three seeds (0.80, 0.25, 0.63). The earlier design let 1/r pass yukawa (0.077).
- **Open issues (need a decision before Phase 2 is checked)**:
  - yukawa (λ = 2) and fractional (1/r²) are indistinguishable on every design tried: each world's
    true law passes the other (nMSE ≈ 0.000–0.002).
  - extra_dimensions: 1/r passes (the crossover only shows within r ≈ 0.5, which the cases exclude
    to avoid near-singular passes). The vendor's own default cases have the same blind spot.
- **Vendor bug found**: `coulomb_easy` (nbody) is *repulsive* with a = |p1|/r², and p2 has no
  effect: the probe's force charge is fixed at 1, so its source charge −|p2| is never used.
  The docstring and mission describe an attractive F = k·p1·p2/r². The oracle scores against the
  simulator as it behaves; vendor code is not edited.
- **Result**: ⏳ not yet checked; no Phase 2 tests committed yet.

## Phase 4: ARA import and first real-data replay market (2026-10-03)
- **Built**:
  - `dm/importers/ara.py`: downloads `SCOREBOARD.md`, the file tree and per-world
    `meta/result/episode.json` for the eight `discoverphysics-<model>-ara` datasets into
    `attempts/cache/ara/<model>/` (git-ignored); runs offline from the cache (`--offline`).
    Scoreboard parsed by header name. One `AttemptRecord` per (model, world), uuid5 id over
    model + world + protocol, written to `attempts/ara.jsonl` (append-only, unchanged records
    skipped). Pass = numeric only (scoreboard nMSE < 0.1); ARA's own verdict kept in `extra`.
    Rounds tie-break: scoreboard, then meta (all three values in `extra.rounds_sources`).
    Non-finite nMSE → `normalised_mse=None, passed=False` (raw string in `extra.nmse_raw`), so
    the store never writes NaN/Infinity. Attribution (ARA Labs, CC BY 4.0) in every record.
  - `dm/importers/vendor_runs.py`: same, for local vendor run JSONs
    (`protocol="discoverphysics_native"`); tested on a hand-made fixture only.
  - `dm/replay.py`: Track A sweep on `ReplayPool(records, rounds_cost(1.0),
    charge_event="round_charged")` (same pool as `cost_model` and `outcome_source`), prizes
    5/20/50/100/200 × seeds 0–4, 200 ticks, 100 credits, leave-one-out starting beliefs.
    Generic clearing-prize rule (lowest prize solved in ≥3 of 5 seeds), per-solver profit/bid
    tables and lab revenue computed from the event log. `--verdict ara` runs the sensitivity
    sweep with ARA's rule.
  - `attempts/fixtures/ara_sample/`: fable, gpt5.5, opus4.8-max (scoreboards + a few worlds).
  - `REPORT_DRAFT.md`.
- **Check**:
  - Counts: 8 models × 11 worlds = 88 records, none missing.
  - nMSE vs `mean_pos_error / _WORLD_VARS` (3 s.f.): disagreements listed below. fable/ether
    has `mean_pos_error = inf` in meta.json and is cross-checked against
    `posthoc_salvage.json` (agrees).
  - Replay: `run_market` conservation plus `sum(compute_final_balances) == 0` in all 25 runs;
    byte-identical `events.json`/`ledger.json`/`summary.json` across two processes;
    no (solver, world) record drawn twice in a run; no agent balance below 0.
  - Tests use only the offline fixture (network calls patched to fail). 142 tests passed
    (100 existing + 42 new).
- **Result**: ✅ PASS. Every world clears at 20 (extra_dimensions at 5) vs 50–200 for the
  stand-ins. **Caveat:** one attempt per (model, world), so outcomes are identical in every
  seed and the ≥3-of-5 rule is close to deterministic. See `REPORT_DRAFT.md`.
- **Data disagreements**:
  - nMSE, systematic: every `coulomb_easy` row (8): scoreboard/recomputed ratio ≈ 2.71, i.e.
    ARA's Var ≈ 4.24 vs vendor 11.465. Scoreboard used. Flips fable/coulomb_easy
    (0.104 fail vs 0.0383 recomputed).
  - nMSE, last digit: gpt5.6-sol/dark_matter (0.0734 vs 0.07347), glm5.2/gravity (1.51e-05 vs
    1.504e-05).
  - Non-finite nMSE: gemini3.1-pro/ether, gemini3.1-pro/three_species (`inf`).
  - Rounds, scoreboard = meta ≠ episode (episode one or two lower): opus4.8-max
    ether (6/6/4), extra_dimensions (14/14/13), oscillator (13/13/12); kimi-k2.7 gravity,
    three_species; gemini3.1-pro hubble, oscillator, three_species, yukawa; glm5.2 circle,
    coulomb_easy, dark_matter, ether, extra_dimensions, fractional, oscillator, yukawa.
  - gemini3.1-pro used 17 rounds on hubble and oscillator (cap is 16).
  - ARA verdict vs numeric-only: 19 records pass numerically but fail ARA's explanation
    threshold (none the other way); listed in `output/ara_import_report.json`.
