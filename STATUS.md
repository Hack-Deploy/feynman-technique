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

## Phase 5: ForceBench venue and offline solvers (branch `phase5/forcebench`)
- **Built**:
  - `dm/wallet.py`: `Wallet(owner, balance)`, credits held as integer milli-credits (so 3 × 0.1
    can spend a 0.3 balance exactly). `charge()` runs before the simulator. It raises
    `InsufficientCredits` after logging an `insufficient_credits` event, and leaves balances unchanged.
  - `dm/venues/forcebench.py`: the 13-launch menu (MDA arXiv 2608.09696 **v3** App. C Table 5;
    seed launch D0 = action 3, free). One launch per round, priced per launch, budget 8,
    σ = 0.03 (MDA Table 7), vendor nbody executors unchanged. `run_attempt(...) ->
    SubmittedAttempt` writes a deterministic transcript to `attempts/transcripts/forcebench/`.
    `SubmittedAttempt` is a local stand-in until Phase 2 adds it to `dm/types.py` (the module
    imports the real one when present).
  - `dm/solvers/_inference.py` (shared), `bayes_lite.py`, `random_menu.py`: library of 5 families
    (power k/r^p, log k/r, Yukawa k·K1(r/λ)/λ, time-modulated k·cos(ωt+φ)/r, crossover
    k1/r + k2/r²) × 3 charge roles (p1/p2, p1, p1·p2) = 15 models, sign of k free.
    Bounded least squares (grid then `least_squares`, with a batched forward-difference Jacobian).
    Known-σ BIC weights, used as an approximation of MDA's SMC evidence with the e^(−2.5·C_m)
    prior (no Occam prior here). Design: `bayes_lite` = argmax BIC-weighted between-model
    variance of predicted positions / price (MDA Eq. 5); `random_menu` = seeded uniform.
    Both stop when no unused launch's VoI exceeds 12·σ² (the models disagree by less than
    the noise on every remaining launch). Submission is the top model as a KDK-leapfrog
    `discovered_law` (dt 0.01, softening 0.05); it reproduces the fitted predictions to 1e-9.
    **`stated_p_success` = BIC weight of the submitted model**, recorded before submission.
  - `tests/forcebench_local.py`: a local scorer (vendor `Evaluator`, a fresh unmetered executor,
    vendor default cases, nMSE = mean_pos_error / `_WORLD_VARS`; nan/inf counts as a fail), a
    1/r-with-fitted-k baseline (vendor `fit_parameters` on the attempt's paid training data), and
    the 6 × 5 × 2 grid (`uv run python -m tests.forcebench_local`). This is NOT the oracle. The
    final check through `dm.settle` waits for Phase 2.
- **Check**: menu = Table 5; wallet exact-fraction and 1000 × 0.1 conservation; insufficient
  wallet → partial attempt still submitted; charges = experiments × price; conservation in
  every attempt; byte-identical transcripts for the same (world, seed); laws equal the fitted
  integrator for all 15 models (< 0.5 s per call at t = 10); local scoring leaves the wallet
  untouched; AST test that `dm/venues`, `dm/solvers` never import `dm.oracle`. 119 passed,
  1 slow skipped (19 s). Full grid (vendor default cases, not hidden cases):

  | world | MDA | MDA menu-LLM | bayes_lite pass | id. | exp. | p̄ | random_menu pass | id. | exp. | p̄ | 1/r+fit pass (bayes / random data) |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | gravity | 100 | 22 | 100 | 5/5 | 2.0 | 0.53 | 100 | 5/5 | 3.4 | 0.60 | 100 / 100 |
  | yukawa | 100 | 33 | 100 | 4/5 | 2.0 | 0.44 | 100 | 3/5 | 6.0 | 0.58 | 100 / 100 |
  | coulomb_easy | 56 | 22 | 100 | 2/5* | 2.0 | 0.49 | 100 | 2/5* | 4.2 | 0.49 | 0 / 60 |
  | oscillator | 100 | 33 | 100 | 5/5 | 1.0 | 1.00 | 100 | 5/5 | 2.8 | 0.99 | 100 / 100 |
  | fractional | 100 | 56 | 100 | 0/5 | 1.4 | 0.24 | 100 | 0/5 | 4.0 | 0.25 | 100 / 100 |
  | extra_dimensions | 100 | 22 | 100 | n/a | 2.2 | 0.94 | 100 | n/a | 5.2 | 0.76 | 0 / 80 |

  (pass = % of 5 seeds with nMSE < 0.1; id. = top family+role matches the truth;
  *the other coulomb seeds chose crossover/p1 with |k1| < 0.03, k2 ≈ −1, the right law inside
  a nesting family. That tie with power/p1 is why coulomb's p̄ ≈ 0.5.) Wall time per attempt: bayes_lite mean 11.5 s (5–22 s), random_menu mean
  29.4 s (7–62 s); a local score takes 0.3–1.4 s.
- **Result**: ✅ PASS for the offline checks. The `dm.settle` (hidden-case) check is pending Phase 2.
- **Findings**:
  - The pass threshold is weak on these cases. A plain 1/r law with fitted k passes yukawa
    (nMSE 0.003–0.056), fractional (0.010–0.069) and oscillator (0.003–0.012) in every seed, and
    extra_dimensions in 4/5 seeds when it is fitted on random_menu's data. So the pass rates on
    those worlds say little about whether a solver found the law. Use the id. column. The
    baseline fails coulomb and extra_dimensions on bayes_lite's data: those attempts are only
    2–3 launches, centred on r0 = 1.5 and p1 = p2 = 2.
  - The vendor `coulomb_easy` world is **repulsive**, with a = |p1|/r² and no effect from p2 (the
    probe's force charge is fixed at 1). This contradicts its docstring and mission text.
    Not fixed; we score against the simulator.
  - Vendor softening: force magnitude uses r_eff = √(r² + 0.05²), direction uses unsoftened r;
    the solvers match.
  - The library cannot represent extra_dimensions (KK image sum) exactly: crossover
    approximates it. fractional (exactly 0.16/r², i.e. power p = 2) is never identified.
    bayes_lite picks Yukawa (λ ≈ 1.3–1.6) in 5/5 seeds, with low stated p (0.12–0.44). It stops
    after 1–2 launches because no menu launch separates these models by more than the noise.
    random_menu's picks are spread across families (p̄ 0.25).
