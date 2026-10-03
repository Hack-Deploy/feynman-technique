# PLAN: Discovery Market on real DiscoverPhysics and ForceBench attempts

Branch: `real-attempts`. One commit per phase; `STATUS.md` gets built / check / result
after each. Standing rules are in `CLAUDE.md`.

## 1. Conflicts between the task prompt and what is in the code (verified 3 Oct 2026)

| # | Prompt says | Found | Resolution |
|---|---|---|---|
| C1 | `.gitignore` has conflict markers; `STATUS.md` stops at P1 | Both fixed in PR #1 (`stage-0-cleanup`, merged) | Phase 0 still replaces `.gitignore` with the short version + new rules |
| C2 | Keep `webpage.html` and `slides.html` | `webpage.html` was deleted upstream (9d50d35); `slides.html`, `slides.pdf` exist | Keep `slides.*`; nothing to restore |
| C3 | Per-world normalising variances come from a results dir not in the repo | Hard-coded as `_WORLD_VARS` in `vendor/.../scripts/run_benchmark.py`. `np.var` over the flattened noise-free test-trajectory coordinates (default test cases, scored particles) reproduces them exactly (gravity 4.283, yukawa 5.677, coulomb_easy 11.465, fractional 5.721, oscillator 6.332, extra_dimensions 4.248) | The oracle uses that same statistic on its *hidden* cases; a test checks it reproduces `_WORLD_VARS` on the default cases |
| C4 | `get_world(name, engine="nbody", …)` | Default engine is `"field"` | Always pass `engine="nbody"` (the vendor runner's default) |
| C5 | MDA Appendix C, Tables 5, 6, 8 | Only in arXiv 2608.09696 **v1–v3**. v4/v5 replaced Appendix C with GlucoseBench; Tables 5/6/8 are now NeuronBench | Cite **v3** everywhere; store this with the menu and Table 8 numbers |
| C6 | DiscoverPhysics noise 0.075 | Vendor runner: absolute σ = 0.075. ARA runs: `noise_frac 0.075` → σ = 0.075·√Var(world) (gravity σ = 0.1552) | DP venue uses absolute 0.075 (configurable `noise_std`); ARA records keep their own σ and are already labelled protocol `ara_harness` |
| C7 | ARA does not record experiment counts reliably | `episode.json` holds per-round `experiment_input` lists (gravity: 36 experiments in 14 rounds) | Record `experiments` when `episode.json` is available; Track A replay still charges rounds × 1 as specified, so it is comparable with the stand-in Track A |
| C8 | ARA verdicts can only be read | `result.json` contains the submitted law, and `episode.json` the training data | Phase 4 as specified (their verdict + numeric-only verdict from their nMSE). Re-scoring with our oracle is an optional extra (`--rescore`), never the default |
| C9 | `score(prereg, law_source)` | Vendor `Evaluator.evaluate` fits `fit_parameters()` on the solver's training trajectories (worlds in `_FIT_WORLDS`) | `score(prereg, law_source, training=None)`; the trajectories are part of the submitted attempt (data the solver paid for), never oracle data |
| C10 | Oracle timeout 120 s | The vendor fit alone may take up to `FIT_TIME_BUDGET_S = 180 s` | Keep 120 s default as specified; every timeout is recorded with its reason. If a real attempt times out during a fit, raise it with the user (it changes what counts as a pass) |
| C11 | Phase 1: emit `confidence_stated` *and* keep every existing output byte-identical | Emitting a new event changes `events.json` | `MarketRun.state_confidence` flag, off for the legacy Track A/C runner, on for replay/live markets. Byte-identity is checked against the Phase 0 baseline (Phase 0 deliberately nulls `outcome.metric`) |
| C12 | Replay "without replacement within a run, with replacement across seeds"; H2 clearing = solved in ≥3 of 5 seeds | ARA has one attempt per (model, world) (seed 0) | Each ARA solver can attempt each world at most once per run; outcomes are fixed, only bid order varies by seed. H2 on ARA is reported with that caveat |
| C13 | ARA: 8 datasets | Dataset README lists 6 siblings (fable, gpt5.6-sol, opus4.8-max, gpt5.5, kimi-k2.7, gemini3.1-pro); glm5.2 and grok4.5 also resolve | Import whatever exists; record counts per model |
| C14 | `coulomb` (repo tables) | Vendor world is `coulomb_easy` | Map `coulomb ↔ coulomb_easy`; `extra dims ↔ extra_dimensions` |

## 2. Files

**Changed:** `.gitignore`, `pyproject.toml` (vendor path deps, scipy, requests), `README.md`,
`STATUS.md`, `market.py` (`CostModel.expected_cost`, `OutcomeSource`, new event fields,
`state_confidence`), `runner.py` (`BernoulliTable` wiring only).

**Added:**
```
vendor/discovery-agents/        submodule @ 450818fa
dm/
  types.py                      Preregistration, AttemptRecord, InsufficientCredits
  store.py                      append-only JSONL attempt store (attempts/*.jsonl)
  wallet.py                     Wallet/Ledger accounts used by venues and markets
  outcomes.py                   OutcomeSource, BernoulliTable, ReplayPool
  oracle/__init__.py            make_prereg, score, explain_score (only settle.py imports it)
  oracle/cases.py               hidden test-case generation per world
  oracle/_worker.py             subprocess entry: evaluate one law, no network
  settle.py                     prereg + submitted attempt → AttemptRecord with verdict
  venues/discoverphysics.py     MeteredExecutor, run_attempt, LLM usage capture
  venues/forcebench.py          13-launch menu, one launch per round, σ = 0.03
  solvers/random_menu.py
  solvers/bayes_lite.py         force-law library × charge roles, LSQ + BIC + VoI/price
  solvers/llm_menu.py           live only
  importers/ara.py              ARA scoreboards/meta → AttemptRecords
  importers/vendor_runs.py      local vendor run JSONs → AttemptRecords
  replay.py                     replay-market sweeps over pools
  calibration.py                Brier + reliability tables
  live_market.py                small real-time market → output/live_events.jsonl
  cli.py                        python -m dm.cli {sim,import-ara,replay,live,preflight,score}
  testing/fake_llm.py           scripted LLM (one experiment, then the 1/r law)
attempts/fixtures/              small committed pools used by tests
tests/test_dm_*.py
dashboard.html, REPORT.md, REPORT_DRAFT.md, DEMO.md
```

## 3. Interfaces as implemented

```python
# dm/types.py — as in the task, plus:
@dataclass(frozen=True)
class Preregistration:
    question_id: str; venue: str; world: str; test_seed: int
    test_cases: list[dict]; metric: str = "normalised_mse"; threshold: float = 0.1
    norm_variance: float; oracle_version: str; public_tests: bool = False
    def commitment(self) -> str  # sha256(canonical JSON: sort_keys, no spaces, floats repr)
    def public(self) -> dict     # everything except test_cases, plus the commitment

AttemptRecord  # exactly the task's fields; to_dict()/from_dict(); JSON-serialisable
class InsufficientCredits(RuntimeError)

# dm/outcomes.py
class OutcomeSource(Protocol):
    def draw(self, agent, world, rng) -> tuple[bool, dict, str | None]  # passed, cost_detail, attempt_id
    def available(self, agent, world) -> bool                            # empty pool → cannot bid
    def stated_p(self, agent, world) -> float | None
class BernoulliTable  # today's behaviour: draws cost via CostModel, pass via rng.random() < p
class ReplayPool      # AttemptRecords by (solver, world); without replacement within a run

# market.py
class CostModel(Protocol): draw_cost, charge_event_type, max_cost, expected_cost(agent, world)
MarketRun(..., outcome_source: OutcomeSource | None = None, state_confidence: bool = False,
          preregs: dict[str, Preregistration] | None = None)

# dm/oracle
make_prereg(venue: str, world: str, test_seed: int) -> Preregistration
score(prereg, law_source: str, training: list | None = None, timeout_s: float = 120) -> dict
  # {normalised_mse, mean_pos_error, passed, prereg_commitment, reason|None, explanation_score: None}
explain_score(world, explanation) -> float | None   # only with ENABLE_LIVE=1

# dm/settle.py — the only importer of dm.oracle
settle(prereg, submitted: SubmittedAttempt) -> AttemptRecord

# dm/venues/discoverphysics.py
class MeteredExecutor: run(list[dict]) -> list[dict]  # charges len × price first, raises InsufficientCredits
run_attempt(solver_model, world, seed, wallet, price, market_aware=True, llm=None) -> SubmittedAttempt
```

Event types added: `prereg_committed`, `prereg_revealed`, `confidence_stated`,
`attempt_started`, `experiment_charged` (+`count`), `attempt_submitted`, `verdict_issued`,
`insufficient_credits`. New optional `Event` fields (`count`, `p`, `commitment`,
`attempt_id`, `detail`) are serialised only when set, so legacy outputs keep their bytes.

## 4. Oracle test cases

* Two-particle worlds (gravity, yukawa, coulomb_easy, oscillator, fractional,
  extra_dimensions): from `test_seed`, draw a base launch r0 ∈ U[2.5, 5], tangential speed
  ∈ U[0.2, 0.5]. Cases: one long-horizon probe (p1 = p2 = 1, t = 1..10) plus single-knob
  interventions p1 ∈ {3, 4, 5} (p2 = 1) and p2 ∈ {3, 5} (p1 = 1), t = 0.5..5. Reject and
  redraw (seeded) any case whose noise-free trajectory comes within r < 0.5 of the source.
* Multi-particle worlds: same keys as the world's default cases, initial conditions
  perturbed from the test seed. A world whose evaluator rejects them falls back to defaults
  with `public_tests=True`, shown in every output.
* `norm_variance = np.var(flattened noise-free scored positions)` — C3.

## 5. Order of work

Phase 0 → 1 → 2 → 3 → 4 → **STOP 1** (ask for the cheapest model and a budget) → 5 → 6 (only
within the approved budget) → 7 → 8. Paid calls only with `ENABLE_LIVE=1` and `DM_MAX_USD`.

## 6. Bounty benchmark on DiscoverPhysics (`poc/`, 3 Oct 2026)

The market from `slides.html` on the 11 DiscoverPhysics worlds: a posted hypothesis with
resolution criteria and a prize; the AI scientist bids or walks away, pays per round and per
experiment, states its assessment, p_success and planned cost every round, and is paid only for a
clear verdict that matches the hidden answer. Failed runs (with their data, without their
conclusions) are shown to later runs; no other memory. All settings in `poc/config.yaml`;
details in `poc/README.md`. Vendor code is not edited; the vendor agent loop is re-implemented in
`poc/agent.py` because it cannot be extended from outside.

**Before a paid run:** the offline `tests/test_poc.py` suite runs and passes. Pick models and
seeds, run the spend preflight, and get an approved API budget (STOP 1).
