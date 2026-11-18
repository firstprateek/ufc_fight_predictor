# Baseline backtest

- snapshot: `96adace863c0a793579cc06d27d468797545607f`
- command: `uv run ufc backtest`
- test period: 2015-01-03 to 2026-11-14 (start 2015-01-01), 5845 scored fights (win outcomes only), 495 events
- features rows: 9000
- each quarter: fit on all scored fights dated before the quarter, predict the quarter
- ECE: 10 equal-width bins, each fight counted from both fighters' views
- accuracy: p = 0.5 counts as half a hit
- models train only on fights where neither fighter has a filled-in snapshot value (height, reach, stance, dob); `snapshot_imputed` flags are never model inputs
- scored fights in the test period carrying a flag: 260 of 5845
- decided fights excluded from training, all history: 1163 of 8843
- last training fold (2026-10-01): 1139 of 8754 fights excluded

## History

Same snapshot and test period throughout. The tables in "Current" below replace both of these.

### Before any fix (first baseline)

The first baseline fed the models the `fighters` snapshot's missingness (reach, stance, height, date of
birth missing for fighters who left the UFC early; see principle 6 in `docs/design.md`). The tree
model's log loss in 2015 was 0.5897 and drifted to about 0.65 later.

| model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| coin_flip | 5845 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| elo | 5845 | 0.6849 | 0.2459 | 0.557 | 0.0188 |
| logreg | 5845 | 0.6601 | 0.2340 | 0.607 | 0.0148 |
| hgb | 5845 | 0.6482 | 0.2287 | 0.624 | 0.0226 |

### After imputation (PR #6)

Missing physical fields filled with constants (median height of the division, reach = height x ratio,
Orthodox, median age), models still trained on every fight. This left a residual leak: imputed values
are exact constants, so a tree could isolate them (fights where exactly one fighter had the imputed
age: that fighter won 16.4% of 165). Tree-model log loss by year: 2015 0.6663, 2016 0.6550, 2017
0.6475, 2018 0.6595, 2019 0.6790, 2020 0.6545, 2021 0.6609, 2022 0.6478, 2023 0.6644, 2024 0.6485,
2025 0.6408, 2026 0.6457.

| model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| coin_flip | 5845 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| elo | 5845 | 0.6849 | 0.2459 | 0.557 | 0.0188 |
| logreg | 5845 | 0.6603 | 0.2341 | 0.607 | 0.0138 |
| hgb | 5845 | 0.6558 | 0.2318 | 0.617 | 0.0120 |

## Current: after training exclusion (this PR)


### Overall

| model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| coin_flip | 5845 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| elo | 5845 | 0.6849 | 0.2459 | 0.557 | 0.0188 |
| logreg | 5845 | 0.6610 | 0.2344 | 0.606 | 0.0169 |
| hgb | 5845 | 0.6583 | 0.2330 | 0.610 | 0.0127 |

### Log loss minus Elo (paired bootstrap over events, 1000 draws, 95%)

Negative means better than Elo.

| model | difference | 95% interval |
|---|---|---|
| coin_flip | +0.0083 | [+0.0060, +0.0105] |
| logreg | -0.0238 | [-0.0286, -0.0187] |
| hgb | -0.0266 | [-0.0326, -0.0207] |

### Calibration of the best model (hgb, both fighter views)

| p bin | fighter views | mean predicted | observed win rate |
|---|---|---|---|
| 0.0-0.1 | 1 | 0.077 | 0.000 |
| 0.1-0.2 | 45 | 0.175 | 0.289 |
| 0.2-0.3 | 586 | 0.264 | 0.273 |
| 0.3-0.4 | 1953 | 0.356 | 0.343 |
| 0.4-0.5 | 3258 | 0.452 | 0.440 |
| 0.5-0.6 | 3262 | 0.548 | 0.560 |
| 0.6-0.7 | 1953 | 0.644 | 0.657 |
| 0.7-0.8 | 586 | 0.736 | 0.727 |
| 0.8-0.9 | 45 | 0.825 | 0.711 |
| 0.9-1.0 | 1 | 0.923 | 1.000 |

### Per year

| year | model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|---|
| 2015 | coin_flip | 464 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2015 | elo | 464 | 0.6856 | 0.2463 | 0.560 | 0.0290 |
| 2015 | logreg | 464 | 0.6648 | 0.2367 | 0.591 | 0.0106 |
| 2015 | hgb | 464 | 0.6734 | 0.2397 | 0.599 | 0.0342 |
| 2016 | coin_flip | 483 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2016 | elo | 483 | 0.6772 | 0.2422 | 0.575 | 0.0335 |
| 2016 | logreg | 483 | 0.6582 | 0.2328 | 0.623 | 0.0411 |
| 2016 | hgb | 483 | 0.6695 | 0.2379 | 0.611 | 0.0285 |
| 2017 | coin_flip | 446 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2017 | elo | 446 | 0.6868 | 0.2468 | 0.572 | 0.0360 |
| 2017 | logreg | 446 | 0.6633 | 0.2354 | 0.596 | 0.0208 |
| 2017 | hgb | 446 | 0.6456 | 0.2271 | 0.600 | 0.0477 |
| 2018 | coin_flip | 469 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2018 | elo | 469 | 0.6851 | 0.2460 | 0.537 | 0.0072 |
| 2018 | logreg | 469 | 0.6640 | 0.2356 | 0.614 | 0.0310 |
| 2018 | hgb | 469 | 0.6560 | 0.2320 | 0.614 | 0.0294 |
| 2019 | coin_flip | 506 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2019 | elo | 506 | 0.6956 | 0.2512 | 0.501 | 0.0393 |
| 2019 | logreg | 506 | 0.6886 | 0.2477 | 0.583 | 0.0629 |
| 2019 | hgb | 506 | 0.6787 | 0.2428 | 0.589 | 0.0210 |
| 2020 | coin_flip | 444 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2020 | elo | 444 | 0.6832 | 0.2451 | 0.590 | 0.0582 |
| 2020 | logreg | 444 | 0.6687 | 0.2376 | 0.592 | 0.0116 |
| 2020 | hgb | 444 | 0.6571 | 0.2319 | 0.615 | 0.0326 |
| 2021 | coin_flip | 497 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2021 | elo | 497 | 0.6830 | 0.2450 | 0.568 | 0.0313 |
| 2021 | logreg | 497 | 0.6687 | 0.2378 | 0.602 | 0.0266 |
| 2021 | hgb | 497 | 0.6650 | 0.2363 | 0.589 | 0.0156 |
| 2022 | coin_flip | 506 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2022 | elo | 506 | 0.6828 | 0.2448 | 0.577 | 0.0401 |
| 2022 | logreg | 506 | 0.6609 | 0.2347 | 0.603 | 0.0210 |
| 2022 | hgb | 506 | 0.6560 | 0.2322 | 0.609 | 0.0120 |
| 2023 | coin_flip | 504 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2023 | elo | 504 | 0.6869 | 0.2469 | 0.546 | 0.0145 |
| 2023 | logreg | 504 | 0.6586 | 0.2334 | 0.589 | 0.0325 |
| 2023 | hgb | 504 | 0.6655 | 0.2365 | 0.579 | 0.0239 |
| 2024 | coin_flip | 513 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2024 | elo | 513 | 0.6840 | 0.2455 | 0.547 | 0.0075 |
| 2024 | logreg | 513 | 0.6574 | 0.2330 | 0.581 | 0.0396 |
| 2024 | hgb | 513 | 0.6480 | 0.2283 | 0.608 | 0.0189 |
| 2025 | coin_flip | 513 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2025 | elo | 513 | 0.6820 | 0.2445 | 0.560 | 0.0231 |
| 2025 | logreg | 513 | 0.6413 | 0.2251 | 0.632 | 0.0352 |
| 2025 | hgb | 513 | 0.6405 | 0.2245 | 0.669 | 0.0687 |
| 2026 | coin_flip | 500 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2026 | elo | 500 | 0.6861 | 0.2465 | 0.555 | 0.0239 |
| 2026 | logreg | 500 | 0.6396 | 0.2240 | 0.660 | 0.0536 |
| 2026 | hgb | 500 | 0.6447 | 0.2267 | 0.642 | 0.0354 |

### Scored fights with a filled-in snapshot value (flagged) versus without

Models never train on flagged fights; they are still predicted and scored.

| model | subset | fights | log loss | accuracy |
|---|---|---|---|---|
| logreg | flagged | 260 | 0.6975 | 0.538 |
| logreg | unflagged | 5585 | 0.6593 | 0.609 |
| hgb | flagged | 260 | 0.7026 | 0.552 |
| hgb | unflagged | 5585 | 0.6562 | 0.613 |

