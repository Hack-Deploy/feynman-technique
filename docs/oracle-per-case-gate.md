# Sanity gate: per-case nMSE (threshold 0.1)

Context: this table records the per-case scoring evaluation behind the 2026-10-03 decision to leave the judge unchanged.

Hidden cases from `make_prereg('forcebench', world, 0)`. Laws fitted with vendor `fit_parameters` on noisy (σ=0.03) training data, using 3 noise seeds × 2 training regimes (all 13 menu launches; launches 3, 11, 12) = 6 runs per row. coulomb_easy laws use fixed constants (it is not a fitted world). Per-case variance = `np.var` of that case's noise-free scored positions (same convention as `norm_variance`).

| kind | world | law | mean nMSE (range) | max per-case nMSE (range) | pass, pooled | pass, mean | pass, max |
|---|---|---|---|---|---|---|---|
| true | gravity | log/ratio | 5.3e-05–0.00088 | 0.00013–0.0022 | 6/6 | 6/6 | 6/6 |
| true | yukawa | yukawa/ratio | 5.6e-05–0.011 | 0.00017–0.034 | 6/6 | 6/6 | 6/6 |
| true | coulomb_easy | power/p1 k=-1 p=2 (fixed) | 1.4e-14–1.4e-14 | 3.6e-14–3.6e-14 | 6/6 | 6/6 | 6/6 |
| true | oscillator | timemod/ratio | 3.1e-05–0.00056 | 7.2e-05–0.002 | 6/6 | 6/6 | 6/6 |
| true | fractional | power/ratio | 0.00011–0.014 | 0.00034–0.042 | 6/6 | 6/6 | 6/6 |
| true | extra_dimensions | KK image sum/ratio | 0.0009–0.011 | 0.0023–0.037 | 6/6 | 6/6 | 6/6 |
| wrong | oscillator | log/ratio (1/r) | 0.0086–0.012 | 0.019–0.041 | 6/6 | 6/6 | 6/6 |
| wrong | extra_dimensions | log/ratio (1/r) | 0.0095–0.63 | 0.053–1.4 | 3/6 | 3/6 | 3/6 |
| wrong | fractional | yukawa/ratio | 0.0023–0.026 | 0.0071–0.079 | 6/6 | 6/6 | 6/6 |
| wrong | yukawa | power/ratio | 0.001–0.029 | 0.0032–0.091 | 6/6 | 6/6 | 6/6 |
| role | gravity | log/p1 | 0.029–0.046 | 0.045–0.094 | 6/6 | 6/6 | 6/6 |
| role | gravity | log/product | 0.44–0.46 | 0.79–0.91 | 0/6 | 0/6 | 0/6 |
| role | gravity | log/p2÷p1 | 1.3–1.8 | 3.3–3.4 | 0/6 | 0/6 | 0/6 |
| role | coulomb_easy | power/ratio k=-1 p=2 (fixed) | 0.012–0.012 | 0.042–0.042 | 6/6 | 6/6 | 6/6 |

Per-case variances (6 hidden cases): oscillator: 5.65, 5.55, 5.60, 5.73, 5.74, 5.76; fractional: 5.59, 5.22, 5.04, 4.87, 5.73, 5.75; extra_dimensions: 4.76, 3.75, 4.15, 4.97, 5.42, 5.56; gravity: 4.75, 3.74, 4.08, 4.89, 5.42, 5.56; yukawa: 5.64, 5.34, 5.20, 5.06, 5.74, 5.76; coulomb_easy: 7.18, 10.43, 12.24, 14.14, 7.18, 7.18

Pooled variances: oscillator 5.67, fractional 5.37, extra_dimensions 5.22, gravity 5.18, yukawa 5.46, coulomb_easy 9.87
