# fable 11-world 正式成绩单 (2026-07-14)

判据: norm_MSE=MSE/Var(world)<0.1 且 explanation>=0.75。格式失败组(ether, extra_dimensions)按补测(salvage)计,
补测=agent物理公式原样+标准积分外壳,详见各目录 posthoc_salvage.json。

| world | rounds | norm_MSE | expl | verdict | cost | note |
|---|---|---|---|---|---|---|
| circle | 10 | 0.000788 | 0.80 | PASS | $60.68 |  |
| coulomb_easy | 8 | 0.104 | 1.00 | FAIL | $31.28 |  |
| dark_matter | 15 | 0.0537 | 0.80 | PASS | $101.56 |  |
| ether | 7 | 0.0225 | 0.90 | PASS | $57.28 | salvage |
| extra_dimensions | 10 | 0.000707 | 0.90 | PASS | $63.89 | salvage |
| fractional | 8 | 2.8e-07 | 0.50 | FAIL | $37.52 |  |
| gravity | 7 | 9.22e-05 | 1.00 | PASS | $38.59 |  |
| hubble | 10 | 0.0114 | 1.00 | PASS | $78.59 |  |
| oscillator | 15 | 0.00106 | 0.90 | PASS | $69.75 |  |
| three_species | 10 | 0.184 | 0.30 | FAIL | $55.39 |  |
| yukawa | 8 | 2.22e-09 | 1.00 | PASS | $52.02 |  |

**8/11 PASS (73%)** · seed 0 · noise_frac 0.075 · max 16 rounds · CC+ARA bridge (fable)