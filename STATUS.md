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
