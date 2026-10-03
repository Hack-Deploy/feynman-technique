# Discovery Market

A three-sided market for verified science: researchers post prizes for answers, labs are paid
for every experiment, and AI solvers pay for their own experiments and win the prize only when
an independent judge verifies the answer. Failed attempts go to a public ledger when a prize closes.

The market runs on two venues built on DiscoverPhysics (simulated worlds with non-standard
physics, so every answer can be checked exactly):

- **DiscoverPhysics**: the original free-form benchmark, many experiments per round.
- **ForceBench**: our wrapper with a fixed menu of 13 probe launches, one launch per round,
  following the Model Discovery Agent paper (arXiv 2608.09696, Appendix C).

## Commands

```bash
uv sync                              # install
uv run pytest -q                     # all tests; must pass before every commit
uv run python run.py                 # simulated market from published tables (no API calls)
uv run python -m dm.cli --help       # live venues, replay market, oracle (once built)
```

## Layout

- `market.py`, `runner.py`, `analysis.py`, `data_loader.py`, `run.py`: original simulated market
  driven by published pass rates (Tracks A and C). Keep working.
- `dm/`: new code. Oracle, venues, solvers, attempt store, importers, replay and live markets.
- `vendor/discovery-agents/`: DiscoverPhysics, pinned git submodule. **Never edit.** Wrap it.
- `data/`: published tables. Never change values; record disagreements in `STATUS.md`.
- `output/`: generated, git-ignored. `attempts/`: attempt store, git-ignored except `fixtures/`.

## Rules that always apply

1. **No paid API calls unless `ENABLE_LIVE=1` and `DM_MAX_USD` are both set.** Tests never call
   an API; they use the scripted fake LLM in `dm/testing/`. Any live command prints its projected
   spend first and stops if it would exceed `DM_MAX_USD`.
2. **Credits are conserved.** Every credit movement is an event; the sum over all accounts is
   identical at the start and end of every run. Fail loudly otherwise.
3. **The oracle is independent of solvers.** Solver code must not import `dm.oracle` or read
   hidden test cases. A test enforces this. Solver-submitted laws run only inside the oracle's
   sandboxed subprocess with a timeout.
4. **Deterministic.** Every run is reproducible from its config and seed. No wall-clock or
   unseeded randomness in outcomes.
5. **Events and attempt records are append-only.** Never rewrite history; supersede with a new record.
6. **Never edit vendor code.** If DiscoverPhysics needs to behave differently, wrap or subclass it
   in `dm/venues/`, and document why.
7. Python 3.12, type hints, small pure functions, no new heavy dependencies without a note in
   `STATUS.md`. numpy, scipy, pandas, pyarrow, jax (via the vendor package) are fine.
8. After each phase: run the tests, append what you built and the check result to `STATUS.md`,
   commit. If a check fails twice, stop and write down what blocked you.
9. Never log API keys or put them in files. Read them from the environment only.

## Attribution

Imported ARA run data (huggingface.co/AgentNativeResearchLab) is CC BY 4.0: credit
"ARA Labs (AgentNativeResearchLab)" with a link wherever it is shown. Use only the 11 public
DiscoverPhysics worlds; do not try to obtain the gated private worlds.
