# codex (gpt-5.5, effort=high) 11-world 正式成绩单 (2026-07-15)

判据与 fable 组一致: norm_MSE = mean_pos_error/Var(world) < 0.1 且 explanation >= 0.75。
Var(world) 与两组 noise_std/seed 逐 world 完全一致 (seed 0, noise_frac 0.075, ≤16 rounds)。
裁判固定 Claude Code 一次性调用 (跨 harness 可比)。
本组 agent.py 含 final_law compile+signature 校验+2次重试 (2026-07-14 改动); fable 官方
11 局早于该改动, 其中 ether/extra_dimensions 按 post-hoc salvage 计。
token 口径: 同 codex-sol 组 (session 累计值见该组说明)。cost 仅 judge 开销, gpt-5.5 本体走订阅。
| world | rounds | norm_MSE | expl | verdict | out_tok | cost | fable对照 |
|---|---|---|---|---|---|---|---|
| circle | 8 | 0.0105 | 0.50 | FAIL | 169k | $0.07 | PASS |
| coulomb_easy | 6 | 469 | 0.90 | FAIL | 119k | $0.06 | FAIL |
| dark_matter | 10 | 0.782 | 0.50 | FAIL | 200k | $0.07 | PASS |
| ether | 6 | 0.15 | 0.50 | FAIL | 106k | $0.06 | PASS |
| extra_dimensions | 11 | 0.00865 | 0.30 | FAIL | 254k | $0.06 | PASS |
| fractional | 8 | 7.03e-05 | 0.50 | FAIL | 157k | $0.05 | FAIL |
| gravity | 14 | 0.000115 | 0.90 | PASS | 141k | $0.07 | PASS |
| hubble | 15 | 1.24 | 0.50 | FAIL | 461k | $0.07 | PASS |
| oscillator | 10 | 0.96 | 0.40 | FAIL | 296k | $0.06 | PASS |
| three_species | 15 | 0.152 | 0.40 | FAIL | 513k | $0.07 | FAIL |
| yukawa | 12 | 0.004 | 0.90 | PASS | 374k | $0.06 | PASS |

**2/11 PASS (18%)** · seed 0 · noise_frac 0.075 · max 16 rounds · 总成本 $1

FAIL 详情: circle (norm_MSE 0.0105✓, expl 0.5✗);
coulomb_easy (norm_MSE 469✗, expl 0.9✓);
dark_matter (norm_MSE 0.782✗, expl 0.5✗);
ether (norm_MSE 0.15✗, expl 0.5✗);
extra_dimensions (norm_MSE 0.00865✓, expl 0.3✗);
fractional (norm_MSE 7.03e-05✓, expl 0.5✗);
hubble (norm_MSE 1.24✗, expl 0.5✗);
oscillator (norm_MSE 0.96✗, expl 0.4✗);
three_species (norm_MSE 0.152✗, expl 0.4✗)。
