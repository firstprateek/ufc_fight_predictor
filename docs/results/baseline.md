# Baseline backtest

- snapshot: `96adace863c0a793579cc06d27d468797545607f`
- command: `uv run ufc backtest`
- test period: 2015-01-03 to 2026-11-14 (start 2015-01-01), 5845 scored fights (win outcomes only), 495 events
- features rows: 9000
- each quarter: fit on all scored fights dated before the quarter, predict the quarter
- ECE: 10 equal-width bins, each fight counted from both fighters' views
- accuracy: p = 0.5 counts as half a hit

### Before the leak fix (first baseline, superseded)

The first baseline fed the models the `fighters` snapshot's missingness (reach, stance, height, date of
birth missing for fighters who left the UFC early; see principle 6 in `docs/design.md`). The tree
model's log loss in 2015 was 0.5897 and drifted to about 0.65 later. Overall, same snapshot and test
period:

| model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| coin_flip | 5845 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| elo | 5845 | 0.6849 | 0.2459 | 0.557 | 0.0188 |
| logreg | 5845 | 0.6601 | 0.2340 | 0.607 | 0.0148 |
| hgb | 5845 | 0.6482 | 0.2287 | 0.624 | 0.0226 |

Everything below is after the fix: physical fields imputed with static rules, no missingness visible to
any model.

### Overall

| model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| coin_flip | 5845 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| elo | 5845 | 0.6849 | 0.2459 | 0.557 | 0.0188 |
| logreg | 5845 | 0.6603 | 0.2341 | 0.607 | 0.0138 |
| hgb | 5845 | 0.6558 | 0.2318 | 0.617 | 0.0120 |

### Log loss minus Elo (paired bootstrap over events, 1000 draws, 95%)

Negative means better than Elo.

| model | difference | 95% interval |
|---|---|---|
| coin_flip | +0.0083 | [+0.0060, +0.0105] |
| logreg | -0.0246 | [-0.0291, -0.0200] |
| hgb | -0.0291 | [-0.0350, -0.0234] |

### Calibration of the best model (hgb, both fighter views)

| p bin | fighter views | mean predicted | observed win rate |
|---|---|---|---|
| 0.1-0.2 | 60 | 0.176 | 0.233 |
| 0.2-0.3 | 691 | 0.264 | 0.252 |
| 0.3-0.4 | 1994 | 0.356 | 0.355 |
| 0.4-0.5 | 3090 | 0.451 | 0.432 |
| 0.5-0.6 | 3110 | 0.549 | 0.567 |
| 0.6-0.7 | 1994 | 0.644 | 0.645 |
| 0.7-0.8 | 691 | 0.736 | 0.748 |
| 0.8-0.9 | 60 | 0.824 | 0.767 |

### Per year

| year | model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|---|
| 2015 | coin_flip | 464 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2015 | elo | 464 | 0.6856 | 0.2463 | 0.560 | 0.0290 |
| 2015 | logreg | 464 | 0.6633 | 0.2361 | 0.586 | 0.0104 |
| 2015 | hgb | 464 | 0.6663 | 0.2363 | 0.607 | 0.0128 |
| 2016 | coin_flip | 483 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2016 | elo | 483 | 0.6772 | 0.2422 | 0.575 | 0.0335 |
| 2016 | logreg | 483 | 0.6556 | 0.2319 | 0.621 | 0.0542 |
| 2016 | hgb | 483 | 0.6550 | 0.2313 | 0.631 | 0.0334 |
| 2017 | coin_flip | 446 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2017 | elo | 446 | 0.6868 | 0.2468 | 0.572 | 0.0360 |
| 2017 | logreg | 446 | 0.6665 | 0.2370 | 0.592 | 0.0217 |
| 2017 | hgb | 446 | 0.6475 | 0.2282 | 0.605 | 0.0207 |
| 2018 | coin_flip | 469 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2018 | elo | 469 | 0.6851 | 0.2460 | 0.537 | 0.0072 |
| 2018 | logreg | 469 | 0.6645 | 0.2358 | 0.633 | 0.0621 |
| 2018 | hgb | 469 | 0.6595 | 0.2337 | 0.600 | 0.0213 |
| 2019 | coin_flip | 506 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2019 | elo | 506 | 0.6956 | 0.2512 | 0.501 | 0.0393 |
| 2019 | logreg | 506 | 0.6844 | 0.2456 | 0.591 | 0.0483 |
| 2019 | hgb | 506 | 0.6790 | 0.2429 | 0.581 | 0.0253 |
| 2020 | coin_flip | 444 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2020 | elo | 444 | 0.6832 | 0.2451 | 0.590 | 0.0582 |
| 2020 | logreg | 444 | 0.6664 | 0.2364 | 0.599 | 0.0285 |
| 2020 | hgb | 444 | 0.6545 | 0.2307 | 0.633 | 0.0380 |
| 2021 | coin_flip | 497 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2021 | elo | 497 | 0.6830 | 0.2450 | 0.568 | 0.0313 |
| 2021 | logreg | 497 | 0.6662 | 0.2367 | 0.598 | 0.0174 |
| 2021 | hgb | 497 | 0.6609 | 0.2344 | 0.591 | 0.0159 |
| 2022 | coin_flip | 506 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2022 | elo | 506 | 0.6828 | 0.2448 | 0.577 | 0.0401 |
| 2022 | logreg | 506 | 0.6579 | 0.2332 | 0.613 | 0.0365 |
| 2022 | hgb | 506 | 0.6478 | 0.2282 | 0.625 | 0.0247 |
| 2023 | coin_flip | 504 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2023 | elo | 504 | 0.6869 | 0.2469 | 0.546 | 0.0145 |
| 2023 | logreg | 504 | 0.6578 | 0.2331 | 0.601 | 0.0140 |
| 2023 | hgb | 504 | 0.6644 | 0.2360 | 0.607 | 0.0177 |
| 2024 | coin_flip | 513 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2024 | elo | 513 | 0.6840 | 0.2455 | 0.547 | 0.0075 |
| 2024 | logreg | 513 | 0.6586 | 0.2337 | 0.575 | 0.0549 |
| 2024 | hgb | 513 | 0.6485 | 0.2286 | 0.626 | 0.0261 |
| 2025 | coin_flip | 513 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2025 | elo | 513 | 0.6820 | 0.2445 | 0.560 | 0.0231 |
| 2025 | logreg | 513 | 0.6424 | 0.2257 | 0.630 | 0.0306 |
| 2025 | hgb | 513 | 0.6408 | 0.2245 | 0.665 | 0.0738 |
| 2026 | coin_flip | 500 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2026 | elo | 500 | 0.6861 | 0.2465 | 0.555 | 0.0239 |
| 2026 | logreg | 500 | 0.6417 | 0.2248 | 0.644 | 0.0378 |
| 2026 | hgb | 500 | 0.6457 | 0.2271 | 0.634 | 0.0232 |

