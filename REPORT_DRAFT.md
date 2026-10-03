> Working draft; there is no separate `REPORT.md`.
> Numbers below come from the committed fixture snapshots.

# Discovery Market: first replay on real attempts (draft)

Data: the eight published ARA DiscoverPhysics runs by **ARA Labs (AgentNativeResearchLab)**,
<https://huggingface.co/AgentNativeResearchLab> (datasets
`discoverphysics-<model>-ara`), licensed CC BY 4.0. Imported unchanged apart from the
normalisation described below; the per-record attribution is in each `AttemptRecord.extra`.

Reproduce (offline once `attempts/cache/ara/` is populated):

```
uv run python -m dm.importers.ara          # → attempts/ara.jsonl, output/ara_import_report.json
uv run python -m dm.replay ara             # → output/replay_ara/
uv run python -m dm.replay ara --verdict ara   # sensitivity → output/replay_ara_araverdict/
```

## Setup

- 88 records = 8 models × 11 worlds, one attempt each (`source="published_replay"`,
  `protocol="ara_harness"`, seed 0, ≤16 rounds, σ = 0.075·√Var(world)).
- Market: Track A, `ReplayPool(records, rounds_cost(1.0), charge_event="round_charged")` used as
  both `cost_model` and `outcome_source`; prizes 5, 20, 50, 100, 200; seeds 0–4; 200 ticks;
  100 starting credits; `true_probs=None`, `state_confidence=True`.
- Cost of an attempt = its recorded rounds × 1 credit (paid to the lab).
- Starting belief of solver *s* about world *w* = *s*'s pass rate on the other 10 worlds
  (leave-one-out). This never uses the record being bid on, but it is still in-sample: it is
  computed from the same 88 records the market replays.
- Pass rule (main result): numeric only, nMSE < 0.1 using the scoreboard nMSE. ARA's own rule
  (nMSE < 0.1 **and** explanation ≥ 0.75) is shown as a sensitivity run.
- Stand-in comparison: `output/summary.json` `h2_clearing_prizes` (Track A on Bernoulli stand-in
  probabilities, raw / calibrated / power odds). ARA's `coulomb_easy` is compared with the
  stand-in `coulomb`.

> **Caveat (applies to every H2 number below).** ARA has exactly one attempt per
> (model, world). Within a run each solver can try each world at most once, and a world's
> outcome is the same in every seed; only the order of bids changes. The "solved in ≥ 3 of 5
> seeds" clearing rule is therefore close to deterministic here: in both sweeps no
> (prize, world) cell was solved in some seeds but not others. The clearing prizes measure
> "is the prize high enough for some solver with a passing record to bid", not a success
> frequency.

### Clearing prizes

| world | ARA replay (numeric nMSE<0.1) | ARA replay (ARA rule, sensitivity) | stand-in raw | stand-in calibrated | stand-in power |
|---|---|---|---|---|---|
| circle | 20 | 20 | 50 | 100 | 200 |
| coulomb_easy | 20 | 20 | 50 | 50 | 50 |
| dark_matter | 20 | 50 | 100 | 100 | 200 |
| ether | 20 | 20 | 50 | 100 | 100 |
| extra_dimensions | 5 | 20 | 50 | 100 | 200 |
| fractional | 20 | 20 | 100 | 100 | 100 |
| gravity | 20 | 20 | 50 | 50 | 50 |
| hubble | 20 | 20 | 50 | 100 | 50 |
| oscillator | 20 | 50 | 50 | 100 | 100 |
| three_species | 20 | 20 | 100 | 100 | never |
| yukawa | 20 | 20 | 50 | 50 | 50 |

Same caveat: one attempt per (model, world), outcomes identical across seeds, so the
≥3-of-5 rule is close to deterministic.

Reading: on the real records every world clears at 20 (numeric rule), against 50–200 for the
stand-ins. Two things drive this. (1) The best real solvers have high priors (gpt5.6-sol 0.9–1.0,
fable and grok4.5 0.8–0.9) and an attempt costs at most 16 credits, so belief × 20 already
exceeds the cost on most worlds. (2) Every world has at least one passing record under the
numeric rule (the stand-ins have worlds that are hard for everyone). `extra_dimensions` clears
at 5 because grok4.5 passes it in 3 rounds. Under ARA's stricter rule, `dark_matter` and
`oscillator` rise to 50 and `extra_dimensions` to 20; nothing becomes unsolvable, because
gpt5.6-sol or fable holds an ARA-passing record on every world.

### Record outcomes (8 models × 11 worlds; P = passes the rule, rounds in brackets)

| world | fable | gemini3.1-pro | glm5.2 | gpt5.5 | gpt5.6-sol | grok4.5 | kimi-k2.7 | opus4.8-max |
|---|---|---|---|---|---|---|---|---|
| circle | P/P (10) | ·/· (3) | P/· (13) | P/· (8) | P/P (14) | P/· (5) | ·/· (13) | P/· (10) |
| coulomb_easy | ·/· (8) | ·/· (3) | ·/· (6) | ·/· (6) | P/P (11) | ·/· (4) | ·/· (9) | ·/· (8) |
| dark_matter | P/P (15) | ·/· (15) | ·/· (16) | ·/· (10) | P/· (14) | P/· (4) | ·/· (6) | ·/· (15) |
| ether | P/P (7) | ·/· (4) | ·/· (10) | ·/· (6) | ·/· (15) | ·/· (6) | ·/· (9) | ·/· (6) |
| extra_dimensions | P/P (10) | ·/· (5) | ·/· (15) | P/· (11) | P/· (12) | P/· (3) | P/· (5) | P/· (14) |
| fractional | P/· (8) | ·/· (5) | P/· (12) | P/· (8) | P/P (8) | P/P (7) | P/· (15) | P/· (10) |
| gravity | P/P (7) | P/P (4) | P/P (15) | P/P (14) | P/P (5) | P/P (4) | P/· (8) | P/P (7) |
| hubble | P/P (10) | ·/· (17) | P/P (11) | ·/· (15) | P/P (9) | P/P (4) | ·/· (8) | ·/· (15) |
| oscillator | P/P (15) | ·/· (17) | ·/· (9) | ·/· (10) | P/P (15) | P/· (7) | ·/· (15) | ·/· (13) |
| three_species | ·/· (10) | ·/· (6) | ·/· (15) | ·/· (15) | P/P (7) | P/P (7) | ·/· (15) | ·/· (15) |
| yukawa | P/P (8) | P/· (7) | P/P (10) | P/P (12) | P/P (12) | P/P (6) | P/P (8) | P/P (6) |
| **passes** | **9/8** | **2/1** | **5/3** | **5/2** | **10/8** | **9/5** | **4/1** | **5/2** |

Cell = numeric rule / ARA rule. **19** of the 88 records pass the numeric rule but fail ARA's
own verdict (explanation < 0.75); none go the other way. They are listed in
`output/ara_import_report.json` and STATUS.md.

## H1-style tables

Same caveat: one attempt per (model, world), so the sd below comes only from bid order
(which solver reaches a world first), not from re-drawn outcomes.

### Profit per solver, numeric-only rule (main result)

Mean ± sd over 5 seeds of final − opening balance; bids = mean `bid_placed` per run; wins = mean prizes won.

| solver | prize 5 | prize 20 | prize 50 | prize 100 | prize 200 |
|---|---|---|---|---|---|
| fable | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +22.4 ± 12.9 (bids 5.0, wins 3.6) | +93.6 ± 44.3 (bids 3.6, wins 2.6) | +207.2 ± 93.0 (bids 3.4, wins 2.4) | +447.2 ± 194.9 (bids 3.4, wins 2.4) |
| gemini3.1-pro | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -1.8 ± 1.5 (bids 0.6, wins 0.0) | -3.2 ± 2.9 (bids 0.8, wins 0.0) | +21.6 ± 49.2 (bids 2.4, wins 0.4) | +61.6 ± 97.9 (bids 2.4, wins 0.4) |
| glm5.2 | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -2.4 ± 2.9 (bids 0.4, wins 0.0) | +17.0 ± 44.1 (bids 2.8, wins 1.0) | +52.6 ± 49.1 (bids 4.0, wins 1.0) | +152.6 ± 110.3 (bids 4.0, wins 1.0) |
| gpt5.5 | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -7.2 ± 4.5 (bids 1.2, wins 0.0) | +18.2 ± 26.1 (bids 4.0, wins 1.2) | +89.6 ± 69.8 (bids 3.6, wins 1.2) | +209.6 ± 144.4 (bids 3.6, wins 1.2) |
| gpt5.6-sol | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +23.4 ± 9.7 (bids 3.8, wins 3.4) | +122.0 ± 82.5 (bids 4.2, wins 3.4) | +225.4 ± 147.0 (bids 3.2, wins 2.6) | +485.4 ± 309.4 (bids 3.2, wins 2.6) |
| grok4.5 | -2.0 ± 0.0 (bids 2.0, wins 1.0) | +37.2 ± 19.8 (bids 4.0, wins 3.0) | +74.8 ± 33.8 (bids 2.8, wins 1.8) | +204.4 ± 126.3 (bids 3.0, wins 2.2) | +424.4 ± 258.9 (bids 3.0, wins 2.2) |
| kimi-k2.7 | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +12.0 ± 6.0 (bids 0.8, wins 0.8) | +0.4 ± 28.1 (bids 1.8, wins 0.4) | +17.2 ± 42.0 (bids 2.4, wins 0.4) | +57.2 ± 90.0 (bids 2.4, wins 0.4) |
| opus4.8-max | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -7.2 ± 7.3 (bids 1.6, wins 0.2) | +0.8 ± 37.9 (bids 2.6, wins 0.6) | +45.8 ± 36.6 (bids 3.2, wins 0.8) | +125.8 ± 75.5 (bids 3.2, wins 0.8) |
| **lab revenue** | **7.0 ± 0.0** | **143.6 ± 16.3** | **226.4 ± 24.6** | **236.2 ± 17.5** | **236.2 ± 17.5** |

### Profit per solver, ARA rule (sensitivity)

Mean ± sd over 5 seeds of final − opening balance; bids = mean `bid_placed` per run; wins = mean prizes won.

| solver | prize 5 | prize 20 | prize 50 | prize 100 | prize 200 |
|---|---|---|---|---|---|
| fable | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +36.8 ± 5.4 (bids 4.6, wins 3.8) | +224.6 ± 29.9 (bids 6.4, wins 5.8) | +437.6 ± 75.7 (bids 6.2, wins 5.0) | +894.0 ± 248.9 (bids 6.4, wins 4.8) |
| gemini3.1-pro | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -3.2 ± 1.9 (bids 1.0, wins 0.0) | -11.8 ± 5.0 (bids 2.4, wins 0.0) | -21.0 ± 11.0 (bids 3.0, wins 0.0) |
| glm5.2 | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -17.6 ± 5.7 (bids 1.8, wins 0.0) | +17.4 ± 51.1 (bids 3.4, wins 0.6) | +63.0 ± 150.3 (bids 4.6, wins 0.6) |
| gpt5.5 | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -16.0 ± 5.4 (bids 2.2, wins 0.0) | -22.2 ± 6.7 (bids 2.4, wins 0.0) | -11.4 ± 74.8 (bids 5.4, wins 0.2) |
| gpt5.6-sol | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +0.4 ± 11.1 (bids 5.2, wins 3.0) | +134.4 ± 41.0 (bids 5.0, wins 3.8) | +261.0 ± 77.6 (bids 5.0, wins 3.2) | +581.0 ± 213.1 (bids 5.0, wins 3.2) |
| grok4.5 | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +13.4 ± 5.5 (bids 5.6, wins 2.2) | +49.8 ± 41.0 (bids 3.8, wins 1.4) | +134.4 ± 47.8 (bids 5.0, wins 1.6) | +298.6 ± 154.7 (bids 4.4, wins 1.6) |
| kimi-k2.7 | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -18.4 ± 5.6 (bids 2.4, wins 0.0) | -41.2 ± 14.0 (bids 4.0, wins 0.0) |
| opus4.8-max | +0.0 ± 0.0 (bids 0.0, wins 0.0) | +0.0 ± 0.0 (bids 0.0, wins 0.0) | -8.4 ± 5.3 (bids 1.2, wins 0.0) | +16.8 ± 52.9 (bids 4.0, wins 0.6) | +75.2 ± 154.6 (bids 4.2, wins 0.6) |
| **lab revenue** | **0.0 ± 0.0** | **129.4 ± 11.3** | **186.4 ± 15.8** | **285.2 ± 27.1** | **361.8 ± 42.8** |

### Starting beliefs (leave-one-out pass rate, range over worlds)

| solver | numeric rule | ARA rule |
|---|---|---|
| fable | 0.80–0.90 | 0.70–0.80 |
| gemini3.1-pro | 0.10–0.20 | 0.00–0.10 |
| glm5.2 | 0.40–0.50 | 0.20–0.30 |
| gpt5.5 | 0.40–0.50 | 0.10–0.20 |
| gpt5.6-sol | 0.90–1.00 | 0.70–0.80 |
| grok4.5 | 0.80–0.90 | 0.40–0.50 |
| kimi-k2.7 | 0.30–0.40 | 0.00–0.10 |
| opus4.8-max | 0.40–0.50 | 0.10–0.20 |

Lab revenue is the total charged for rounds. It stops growing above prize 100 under the
numeric rule because every world is already being attempted and solved early; under the ARA
rule worlds stay open longer, so more failed attempts are paid for.

## Phase 7: H1–H4 from the replay event log

`dm/hypotheses.py` derives each hypothesis from the replay events (`summary.json` →
`hypotheses`); calibration is in `summary.json` → `calibration` (`dm/calibration.py`).
A hypothesis that the pool cannot answer is `UNAVAILABLE` with its reason.

```
uv run python -m dm.replay ara           # ARA pool  → output/replay_ara/
uv run python -m dm.replay forcebench    # ForceBench pool → output/replay_forcebench/
```

`dm.replay forcebench` reads `output/forcebench_settle.json` if present (regenerate with
`uv run python -m tests.forcebench_settle`), else the committed snapshot
`attempts/fixtures/demo/forcebench_settle.json`. The two pools are never mixed: each run
rejects records from another venue. ForceBench setup: 60 settled attempts (6 worlds ×
2 solvers × 5 seeds), `experiments_cost(1.0)` (`experiment_charged`; the first launch is free,
each paid launch costs 1 credit), prizes 2, 5, 10, 20, 50, otherwise as above (seeds 0–4,
200 ticks, 100 credits, leave-one-out beliefs). Credits are conserved in every run of both
sweeps (max |sum| = 0).

Rules: H1 splits solvers at the largest gap in pool pass rate (≥ 0.10 needed) and applies the
Track A rule per prize (strong agents profit, weak agents don't); H2 is the ≥ 3-of-5-seeds
clearing prize; H3 compares pairs whose pass rates differ by ≤ 0.10 and whose paid experiments
per attempt differ by ≥ 0.5, at prizes where both bid; H4 pairs each solver-stated
`confidence_stated` p with its verdict (each source attempt once) and calls the chances
calibrated if |mean p − pass rate| ≤ 0.10.

| | ARA (88 published attempts) | ForceBench (60 settled attempts) |
|---|---|---|
| H1 strong profit, weak stop | **NOT SUPPORTED**. Strong = gpt5.6-sol (10/11), fable, grok4.5 (9/11); gap 0.36 to the rest. At every prize with bids a weak model profits or a strong one doesn't (prize 20: kimi-k2.7 +12.0; prize 50: glm5.2 +17.0, gpt5.5 +18.2) | **UNAVAILABLE**: pass rates 0.967 (bayes_lite) vs 1.0 (random_menu), no strong/weak split |
| H2 clearing prize | **MEASURED**: 20 on every world except extra_dimensions (5); near-deterministic (one attempt per cell) | **MEASURED**: 2 on fractional and oscillator, 5 on the other four worlds |
| H3 fewer experiments earn more | **UNAVAILABLE**: the pool charges rounds; ARA experiment counts are not exact (PLAN C7) | **PARTIAL**: bayes_lite (1.71 paid launches/attempt, pass 0.958) vs random_menu (4.33, 1.0). bayes_lite earns more at prize 5 (11.2 vs 1.6) and 10 (21.4 vs 17.6), not at 20 (49.4 vs 49.6) or 50 (133.4 vs 145.6) |
| H4 calibrated stated chances | **UNAVAILABLE**: no stated p in ARA (88 records excluded); models state their chances on `/live` | **NOT SUPPORTED**: underconfident. Event log: mean p − pass rate = −0.356. All 60 records: mean stated p 0.609, pass rate 0.983, Brier 0.248, calibration-in-the-large −0.374 |

Caveats: the ForceBench pass rates overstate identification on yukawa, fractional, oscillator
and extra_dimensions (lenient hidden tests, Phase 5 findings), so H2/H3 there measure the
current judge; a stricter judge's snapshot can be replayed with the commands above. H3 counts
each source attempt once (24 bayes_lite and 18 random_menu records were drawn across the
sweep). The ARA verdicts are on the numeric rule.

## Data notes

- **Counts**: all eight models present with 11 worlds each (fable, gpt5.6-sol, gpt5.5,
  kimi-k2.7, gemini3.1-pro, opus4.8-max, glm5.2, grok4.5), although the dataset README lists
  only six.
- **nMSE cross-check** (scoreboard vs `mean_pos_error / _WORLD_VARS`, 3 s.f.): 76 of 86 finite
  records agree. All 8 `coulomb_easy` rows disagree by the same factor: ARA's implied
  Var ≈ 4.24 instead of the vendor's 11.465. Two last-digit differences
  (gpt5.6-sol/dark_matter, glm5.2/gravity). The scoreboard value is used. This flips one
  outcome: fable/coulomb_easy is 0.104 on the scoreboard (fail) but 0.038 recomputed (pass).
- **Non-finite nMSE**: gemini3.1-pro/ether and /three_species are `inf` → stored as
  `normalised_mse=None, passed=False`, raw string kept in `extra.nmse_raw`.
- **Rounds**: tie-break = scoreboard, falling back to meta. In 17 records `episode.json` has one
  or two fewer rounds than the scoreboard/meta (opus4.8-max ether/extra_dimensions/oscillator,
  plus kimi-k2.7 ×2, gemini3.1-pro ×4, glm5.2 ×8). All three values are kept in
  `extra.rounds_sources`. gemini3.1-pro used 17 rounds on hubble and oscillator, above the
  stated 16-round cap.
- **Costs**: `llm_usage.usd` is ARA's reported cost. For subscription harnesses it covers only
  the judge. The market charges rounds, not USD.
