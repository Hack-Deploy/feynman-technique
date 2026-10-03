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

## Phase 2: the oracle (2026-10-03)
- **Built**: `dm/oracle/` (`make_prereg`, `score`, `explain_score`), `dm/oracle/cases.py`
  (hidden cases + normalising variance), `dm/settle.py` (the only caller of the oracle),
  `dm.types.SubmittedAttempt`. Review agents added a hidden `test_seed`, the `salt` field,
  `Preregistration.verify`, and an engine guard that refuses replayed verdicts not scored
  against the posted preregistration.
  - **Two-process scoring.** `dm/oracle/_worker.py` (trusted) builds the world, computes the
    noise-free ground truth on the hidden cases, and never runs submitted code.
    `dm/oracle/_sandbox.py` runs the law through the vendor evaluator against a stand-in
    executor whose trajectories are zeros, and returns only predictions; the worker computes
    the errors exactly as the vendor evaluators do. The simulator is never imported in the
    sandbox, and an audit hook refuses imports of `scienceagent`/`physchool`/`jax`, reading
    the vendor tree or `/proc`, subprocesses and network. No API keys or `DM_ORACLE_SECRET`
    reach either subprocess. Best effort: Python in-process restrictions are not a security
    boundary; a determined attacker with arbitrary code could still find a way out (e.g. via
    ctypes into a fresh interpreter). Running the sandbox in a container would close this.
  - **Salt (decision 2026-10-03: secret key in env).** `salt = HMAC-SHA256(DM_ORACLE_SECRET,
    oracle_version|venue|world|test_seed)`, mixed into the case generator and the commitment.
    Same secret → same cases and commitments on every rerun; without the secret, someone
    cannot regenerate candidate cases from the public code and match the published commitment.
    Revealing the salt at close does not reveal the secret. Without `DM_ORACLE_SECRET` the
    cases are exactly as before (unsalted) and a warning is raised. **To use it: set
    `DM_ORACLE_SECRET` to a long random string, keep it out of git, and keep it fixed for the
    life of a market** (changing it changes every hidden case and commitment).
- **Check** (`tests/test_phase2_oracle.py`, 39 tests, about 100 s):
  - 1/r fixture passes gravity (nMSE < 1e-4) and fails yukawa; no law → fail; verdict carries
    the prereg commitment.
  - Hostile laws all fail: infinite loop (timeout), reading the evaluator's stack (was nMSE 0,
    now 4.8), importing and running the true simulator (was nMSE 0, now refused), reading
    vendor source, spawning a process, opening a socket, `sys.exit`, NaN; no secret or API key
    is visible to the law.
  - Sandbox scoring equals the vendor's in-process `mean_pos_error` exactly on all 11 worlds
    (checked by script; 3 worlds in the test suite).
  - Same seed → same commitment; different seeds and different secrets → different cases;
    public view hides cases, seed and salt; reveal verifies; all 5 multi-particle worlds get
    generated (not public) cases; ForceBench refuses multi-particle worlds; normalising
    variance reproduces all 11 vendor `_WORLD_VARS`.
  - Earlier: 60 ForceBench attempts settled through `dm.settle` (Phase 5) with no errors.
- **Result**: ✅ PASS. Full suite 453 passed (490 after Phase 3), 14 xfailed, 4 skipped.
- **Accepted limits of the pass rule (decision 2026-10-03: accept and report).** The rule
  stays nMSE < 0.1 (as in the paper). On these worlds a pass does not prove the right law,
  so "right law" is reported next to "pass" (the `/real` page does this):
  - yukawa (λ = 2) and fractional (1/r²) cannot be told apart on any design tried;
  - extra_dimensions: 1/r passes (the crossover only shows within r ≈ 0.5, excluded to avoid
    near-singular passes; the vendor's defaults have the same blind spot);
  - oscillator: a fitted 1/r passes 10/10 in the Phase 5 settle check.
- **Vendor bug found**: `coulomb_easy` (nbody) is *repulsive* with a = |p1|/r², and p2 has no
  effect; its docstring and mission describe an attractive F = k·p1·p2/r². The oracle scores
  against the simulator as it behaves; vendor code is not edited.
- **Not done**: `explain_score` (live only, needs `ENABLE_LIVE=1`) is untested.

## Phase 3: DiscoverPhysics venue (2026-10-03)
- **Built**:
  - `dm/venues/discoverphysics.py`: `run_attempt(solver_model, world, seed, wallet, price,
    market_aware=True, llm=None, *, prize, max_rounds=16, noise_std=0.075, ask_p, cap)`
    runs the vendor `DiscoveryAgent` unchanged (vendor system prompt, mission, 16 rounds,
    mid-round MSE fit with a per-attempt trajectory CSV in a temp dir) and returns a
    `SubmittedAttempt` for `dm.settle`. All 11 public worlds.
  - `MeteredExecutor`: refuses a batch that is not a list of objects (uncharged); charges
    `len × price` before the simulator runs; `InsufficientCredits` never runs and reaches the
    solver as an experiment error, so it can still submit; if the simulator rejects a paid
    batch (missing key), the charge is refunded (`Wallet.refund`, event
    `experiment_refunded`, lab → agent). Charge events carry the agent's round number.
  - `market_aware` appends the market terms (price, balance, prize) to the mission; the
    system prompt is identical either way.
  - Stated p: after a law is submitted, one more call in the same conversation asks for
    the probability that the law passes (`<p_success>`); unparseable or out of [0, 1] → None.
    Not asked when no law was submitted.
  - `dm/llm.py`: `UsageMeter` (per-attempt calls, tokens, USD; plain-string replies are
    estimated at 4 chars/token and marked `estimated`), `anthropic_llm` (real token counts,
    no server-side fallback so the attempt stays attributed to one model), price table
    (Anthropic first-party, checked 2026-10-03), `SpendCap` (refuses any call whose worst
    case, estimated input + `max_tokens` output, would take the total over `DM_MAX_USD`;
    refuses unpriced models), `live_llm` (needs `ENABLE_LIVE=1` and `DM_MAX_USD`).
  - The meter replaces `scienceagent.llm_client.complete` for the attempt under a lock and
    always restores it (one attempt at a time per process).
  - `dm/testing/fake_llm.py`: scripted LLM (one experiment, then the 1/r law, then p = 0.6).
- **Check** (`tests/test_dm_dp_venue.py`, 37 tests, about 25 s, no API calls): metering
  (charge before run, exact fractional balance 0.3 = 3 × 0.1, non-list uncharged, failed
  batch refunded, noise controls forwarded); fake attempt end to end (2 rounds, 1
  experiment, 0.5 credits, p = 0.6, 3 LLM calls) settles as pass on gravity and fail on
  yukawa; no submission → no law, no p call, settles as fail; insufficient credits shown and
  attempt still submits; market note changes only the mission; deterministic per seed;
  vendor client restored even on provider error; live guard refuses without both variables
  and the cap refuses before calling; vendor tree clean.
- **Result**: ✅ PASS. Full suite 490 passed, 4 skipped, 14 xfailed.
- **Not done**: the cost preflight (projected spend per model × world × seed, printed before
  a live command) is STOP 1 work; no live call has been made.

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

## Phase 4 fix: ARA import under the hardened `AttemptRecord` (2026-10-03)
- **Built**: on top of PR #13's `dm/importers/ara._json_safe` (non-finite floats in `extra` and
  `llm_usage` → `"inf"`/`"-inf"`/`"nan"`), `finite_or_none` keeps `verdict.explanation_score`
  finite or `None` (both importers). `verdict.normalised_mse` was already `None` for non-finite values.
- **Check**: every imported record survives strict `canonical_json` and `from_dict`.
  `uv run python -m dm.importers.ara --offline && uv run python -m dm.replay ara` (plus
  `--verdict ara`) reproduce PR #6's `summary.json` exactly, including clearing prizes.
  A store written by PR #6 holds `Infinity` and no longer loads; regenerate it.
- **Result**: ✅ PASS

## Phase 5: ForceBench venue and offline solvers (branch `phase5/forcebench`)
- **Built**:
  - `dm/wallet.py`: `Wallet(owner, balance)`, credits held as integer milli-credits (so 3 × 0.1
    can spend a 0.3 balance exactly). `charge()` runs before the simulator. It raises
    `InsufficientCredits` after logging an `insufficient_credits` event, and leaves balances unchanged.
  - `dm/venues/forcebench.py`: the 13-launch menu (MDA arXiv 2608.09696 **v3** App. C Table 5;
    seed launch D0 = action 3, free). One launch per round, priced per launch, budget 8,
    σ = 0.03 (MDA Table 7), vendor nbody executors unchanged. `run_attempt(...) ->
    SubmittedAttempt` writes a deterministic transcript to `attempts/transcripts/forcebench/`.
    It returns a `dm.types.SubmittedAttempt` (the Phase 2 type; the earlier local stand-in was
    removed when the branch merged real-attempts).
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
    final check through `dm.settle` is `tests/forcebench_settle.py` (see below).
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
- **Result**: ✅ PASS for the offline checks. Hidden-case check through `dm.settle`: see
  "Settle check" below.
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
- **Settle check, scored via dm.settle (Phase 2 WIP @ aa9928d)**:
  `uv run python -m tests.forcebench_settle` (about 6 min, 6 processes). There is one
  preregistration per world, `prereg_for("forcebench", world, 0)`, from the V1 hidden design:
  tangential launches at r0 ∈ U[3, 6] and t = 1..10, with p1 ∈ {3, 4, 5} or p2 ∈ {3, 5} cases. Each attempt from
  the 6 × 5 × 2 grid goes through `settle(prereg, attempt)`. The 1/r-with-fitted-k baseline is
  settled on the same preregistration, carrying the same attempt's paid `training`. The grid ran
  at a3ab9a2; `dm/oracle`, `dm/settle.py`, `dm/types.py`, venue and solvers are unchanged at aa9928d.
  `prereg_for` and `settle` accept forcebench and match the Phase 2 spec. No worker errors,
  timeouts or non-finite scores; `public_tests` is False everywhere.

  | world | bayes_lite pass | id. | 1/r+fit pass (bayes data) | random_menu pass | id. | 1/r+fit pass (random data) |
  |---|---|---|---|---|---|---|
  | gravity | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
  | yukawa | 5/5 | 4/5 | 0/5 | 5/5 | 3/5 | 1/5 |
  | coulomb_easy | 5/5 | 2/5 | 0/5† | 5/5 | 2/5 | 0/5† |
  | oscillator | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 |
  | fractional | 5/5 | 0/5 | 0/5 | 5/5 | 0/5 | 2/5 |
  | extra_dimensions | 4/5 | 0/5 | 0/5 | 5/5 | 0/5 | 4/5 |

  Solver nMSE ranges:
  - bayes_lite: gravity ≤ 0.0006, yukawa ≤ 0.011, fractional 0.008–0.014,
    extra_dimensions 0.030–0.133 (seed 2 fails).
  - random_menu: fractional 0.002–0.049, extra_dimensions 0.004–0.031.

  Baseline nMSE:
  - yukawa 0.27–0.30 on bayes data (0.016–0.30 on random data).
  - fractional 0.22–0.31 on bayes data (0.058–0.25 on random data).
  - oscillator 0.005–0.042.

  †coulomb_easy is not in the oracle's `FIT_WORLDS` (mirrors the vendor), so the baseline's k
  is never fitted there: it stays at its init 0.1 (nMSE 1.04), and that cell is not a fitted-k
  result. Mean wall time per attempt: 21.4 s (max 66.5 s); settle takes 1.9 s (max 4.5 s).
  - The hidden design fixes most of the threshold problem: 1/r now fails yukawa, fractional
    and coulomb on bayes_lite's data. It still passes oscillator 10/10, and extra_dimensions
    4/5 on random_menu's data. This matches the extra_dimensions blind spot in the Phase 2 notes.
  - fractional still passes with the wrong family: both solvers submit non-power-2 laws
    (mostly Yukawa λ ≈ 1.3–1.6) and pass 5/5. This is the yukawa–fractional degeneracy listed in
    the Phase 2 open issues.

## Phase 8 (partial): "Real attempts" page in the results app (2026-10-03)

- `real.html` at `/real`, fed by `real_data.py` via `GET /api/real`. The original page is now
  labelled "Simulated (published pass rates)" and links to it.
- Run an attempt: `POST /api/real/attempt` runs one ForceBench attempt (world, solver, seed),
  settles it through `dm.settle` on the hidden cases and shows the charges, the purchased data,
  the submitted law, the verdict, whether the law has the true form, and credit conservation.
  Offline, no API key, about 20 s. Uses the existing `_BUSY` lock and host/origin checks.
- ARA replay: reads `attempts/ara.jsonl` and `output/replay_ara/summary.json`. Shows clearing
  prizes next to the simulated ones, profit and bids per model at each prize, lab revenue, and
  all 88 attempts (filterable), with the one-attempt caveat and ARA attribution.
- ForceBench grid: reads `output/forcebench_settle.json`. Shows pass, right-law and 1/r-baseline
  counts per solver and world.
- Missing outputs show a button that runs the generating command (`/api/real/replay`,
  `/api/real/grid`).
- "Right law" uses `identify_model` from `tests/forcebench_local.py` (local reporting only).
- Check: `tests/test_real_app.py` 11 passed with `--runslow`; full suite 387 passed, 4 skipped,
  14 xfailed.

---

# Current state and what is outstanding (2026-10-03, evening)

Demo data: `/real` falls back to committed snapshots in `attempts/fixtures/demo/` (ARA
attempts and replay, ForceBench grid), so it works offline on a fresh clone.
Frontend: the results app now has a "Real attempts" page (`/real`, PR #16) that runs a live
ForceBench attempt through `dm.settle` and shows the ARA replay and the settled ForceBench grid.
`poc/` (bounty benchmark, PR by Stefan) is a separate prototype with its own agent loop.

| Phase | State | Outstanding |
|---|---|---|
| 0 Hygiene and vendor | ✅ done | — |
| 1 Engine generalisation | ✅ done, hardened by review agents | Merge `review/remaining-readiness` (fixes several engine xfails) |
| 2 Oracle | ✅ done | Sandboxed scoring, salt from `DM_ORACLE_SECRET`, 39 tests; pass-rule blind spots accepted and reported; `explain_score` untested (live only) |
| 3 DiscoverPhysics venue | ✅ done offline | No live call yet; `explain_score` at settle untested (live only) |
| 4 ARA import + replay | ✅ done | Caveat: one attempt per (model, world), so H2 is near-deterministic; coulomb_easy nMSE in ARA uses Var ≈ 4.24, not 11.465 |
| STOP 1 Cost preflight | ⏳ next | Preflight command (projected spend from the fake run's token counts × prices); then the user names the cheapest model and approves a budget |
| 5 ForceBench + offline solvers | ✅ done offline and through `dm.settle` | `llm_menu` (live only); pass rates overstate identification on yukawa/fractional/oscillator/extra_dimensions (see Phase 5 findings) |
| 6 Live grid | ⛔ blocked | STOP 1 budget |
| 7 Markets on real attempts | 🟡 partly | ARA replay sweep exists (`dm/replay.py`). Missing: ForceBench and live pools, H1–H4 from the log, H3 on ForceBench, `dm/calibration.py` (Brier + reliability), `dm/live_market.py` |
| 8 Dashboard, report, demo | 🟡 partly | `/real` page in the app (PR #16). Missing: calibration view, replay slider, failure-ledger view, `REPORT.md`, `DEMO.md`, README "three ways to run" |

**Known bugs still marked xfail** (14): float credits in the DP venue spike (3 × 0.1 > 0.3),
mutable verdict dicts inside frozen records, `analysis.compute_clearing_prizes` and
`build_full_summary` assume Track A/C run ids, events can't be joined to ledger rows,
`verdict_issued.commitment` copied rather than checked, cost model vs outcome source can
disagree, negative `lab_cost` accepted, rounds charges have no `count`, dashboard loads
non-cdnjs hosts and drops unknown solver names. Several are fixed on the unmerged
`review/remaining-readiness` branch; rerun the suite after merging it (stale xfails will
XPASS-fail and should be removed).

**Loose ends**: `forcebench_demo.py` (repo root) runs one ForceBench attempt end to end.
`hypothesis` is not installed, so 2 property tests skip.
