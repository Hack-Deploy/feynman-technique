# claude-code (opus-4.8, effort=max) 11-world 正式成绩单 (2026-07-15)

判据与 fable 组一致: norm_MSE = mean_pos_error/Var(world) < 0.1 且 explanation >= 0.75。
Var(world) 与两组 noise_std/seed 逐 world 完全一致 (seed 0, noise_frac 0.075, ≤16 rounds)。
裁判固定 Claude Code 一次性调用 (跨 harness 可比)。
本组 agent.py 含 final_law compile+signature 校验+2次重试 (2026-07-14 改动); fable 官方
11 局早于该改动, 其中 ether/extra_dimensions 按 post-hoc salvage 计。
本组为唯一美元计费组 (与 fable 同 API 口径可直接对比)。salvage 候选 (解释分高/MSE挂, 沿 fable ether 先例, 未补测): ether expl 1.0, coulomb_easy expl 1.0, hubble expl 0.8。
| world | rounds | norm_MSE | expl | verdict | out_tok | cost | fable对照 |
|---|---|---|---|---|---|---|---|
| circle | 10 | 0.000411 | 0.60 | FAIL | 210k | $31.05 | PASS |
| coulomb_easy | 8 | 6.63 | 1.00 | FAIL | 210k | $26.48 | FAIL |
| dark_matter | 15 | 2.65 | 0.30 | FAIL | 212k | $41.97 | PASS |
| ether | 6 | 0.749 | 1.00 | FAIL | 205k | $24.45 | PASS |
| extra_dimensions | 14 | 0.0985 | 0.40 | FAIL | 291k | $53.05 | PASS |
| fractional | 10 | 1.7e-09 | 0.60 | FAIL | 257k | $36.75 | FAIL |
| gravity | 7 | 0.00117 | 1.00 | PASS | 206k | $21.84 | PASS |
| hubble | 15 | 0.954 | 0.80 | FAIL | 342k | $69.09 | PASS |
| oscillator | 13 | 10.3 | 0.20 | FAIL | 319k | $55.76 | PASS |
| three_species | 15 | 0.187 | 0.30 | FAIL | 315k | $62.74 | FAIL |
| yukawa | 6 | 0.000278 | 0.90 | PASS | 208k | $23.70 | PASS |

**2/11 PASS (18%)** · seed 0 · noise_frac 0.075 · max 16 rounds · 总成本 $447

FAIL 详情: circle (norm_MSE 0.000411✓, expl 0.6✗);
coulomb_easy (norm_MSE 6.63✗, expl 1.0✓);
dark_matter (norm_MSE 2.65✗, expl 0.3✗);
ether (norm_MSE 0.749✗, expl 1.0✓);
extra_dimensions (norm_MSE 0.0985✓, expl 0.4✗);
fractional (norm_MSE 1.7e-09✓, expl 0.6✗);
hubble (norm_MSE 0.954✗, expl 0.8✓);
oscillator (norm_MSE 10.3✗, expl 0.2✗);
three_species (norm_MSE 0.187✗, expl 0.3✗)。
