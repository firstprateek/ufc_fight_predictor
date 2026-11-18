# Market benchmark

- odds snapshot: `31d2941509506c9f20e02a5f1de9c253c5289f9c` (`ufc-master.csv`, pre-fight moneyline, book and timing unknown)
- data snapshot: `96adace863c0a793579cc06d27d468797545607f`
- command: `uv run ufc backtest`
- test period: 2015-01-03 to 2026-11-14, scored by the same walk-forward fits as the baseline report

## Against the market

Only fights with odds are scored here: 5179 of 5845 scored fights (88.6%), the same fights for every model. `market` is the odds-implied probability with the bookmaker margin removed (pre-fight moneyline, book and timing unknown); it needs no training.

Sanity: the market favourite won 66.0% of these fights (expected roughly 65%).

### Overall

| model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| coin_flip | 5179 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| elo | 5179 | 0.6847 | 0.2458 | 0.557 | 0.0189 |
| logreg | 5179 | 0.6609 | 0.2344 | 0.606 | 0.0179 |
| hgb | 5179 | 0.6572 | 0.2324 | 0.612 | 0.0134 |
| market | 5179 | 0.6092 | 0.2110 | 0.660 | 0.0168 |

### Log loss minus the market (paired bootstrap over events, 1000 draws, 95%)

Positive means worse than the market.

| model | difference | 95% interval |
|---|---|---|
| coin_flip | +0.0839 | [+0.0748, +0.0933] |
| elo | +0.0755 | [+0.0665, +0.0850] |
| logreg | +0.0517 | [+0.0432, +0.0601] |
| hgb | +0.0480 | [+0.0394, +0.0562] |

### Calibration of the market (both fighter views)

| p bin | fighter views | mean predicted | observed win rate |
|---|---|---|---|
| 0.0-0.1 | 16 | 0.088 | 0.000 |
| 0.1-0.2 | 435 | 0.162 | 0.108 |
| 0.2-0.3 | 1146 | 0.254 | 0.222 |
| 0.3-0.4 | 1754 | 0.351 | 0.355 |
| 0.4-0.5 | 1733 | 0.444 | 0.455 |
| 0.5-0.6 | 1923 | 0.550 | 0.540 |
| 0.6-0.7 | 1754 | 0.649 | 0.645 |
| 0.7-0.8 | 1146 | 0.746 | 0.778 |
| 0.8-0.9 | 435 | 0.838 | 0.892 |
| 0.9-1.0 | 16 | 0.912 | 1.000 |

### Per year

| year | model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|---|
| 2015 | coin_flip | 464 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2015 | elo | 464 | 0.6856 | 0.2463 | 0.560 | 0.0290 |
| 2015 | logreg | 464 | 0.6648 | 0.2367 | 0.591 | 0.0106 |
| 2015 | hgb | 464 | 0.6734 | 0.2397 | 0.599 | 0.0342 |
| 2015 | market | 464 | 0.6340 | 0.2230 | 0.610 | 0.0649 |
| 2016 | coin_flip | 483 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2016 | elo | 483 | 0.6772 | 0.2422 | 0.575 | 0.0335 |
| 2016 | logreg | 483 | 0.6582 | 0.2328 | 0.623 | 0.0411 |
| 2016 | hgb | 483 | 0.6695 | 0.2379 | 0.611 | 0.0285 |
| 2016 | market | 483 | 0.6227 | 0.2170 | 0.630 | 0.0454 |
| 2017 | coin_flip | 440 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2017 | elo | 440 | 0.6864 | 0.2466 | 0.573 | 0.0369 |
| 2017 | logreg | 440 | 0.6626 | 0.2350 | 0.598 | 0.0240 |
| 2017 | hgb | 440 | 0.6451 | 0.2268 | 0.601 | 0.0464 |
| 2017 | market | 440 | 0.6124 | 0.2116 | 0.677 | 0.0299 |
| 2018 | coin_flip | 469 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2018 | elo | 469 | 0.6851 | 0.2460 | 0.537 | 0.0072 |
| 2018 | logreg | 469 | 0.6640 | 0.2356 | 0.614 | 0.0310 |
| 2018 | hgb | 469 | 0.6560 | 0.2320 | 0.614 | 0.0294 |
| 2018 | market | 469 | 0.5999 | 0.2074 | 0.645 | 0.0462 |
| 2019 | coin_flip | 506 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2019 | elo | 506 | 0.6956 | 0.2512 | 0.501 | 0.0393 |
| 2019 | logreg | 506 | 0.6886 | 0.2477 | 0.583 | 0.0629 |
| 2019 | hgb | 506 | 0.6787 | 0.2428 | 0.589 | 0.0210 |
| 2019 | market | 506 | 0.6310 | 0.2210 | 0.635 | 0.0167 |
| 2020 | coin_flip | 443 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2020 | elo | 443 | 0.6832 | 0.2451 | 0.590 | 0.0584 |
| 2020 | logreg | 443 | 0.6690 | 0.2378 | 0.591 | 0.0108 |
| 2020 | hgb | 443 | 0.6573 | 0.2320 | 0.614 | 0.0317 |
| 2020 | market | 443 | 0.6197 | 0.2152 | 0.655 | 0.0296 |
| 2021 | coin_flip | 489 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2021 | elo | 489 | 0.6837 | 0.2453 | 0.563 | 0.0262 |
| 2021 | logreg | 489 | 0.6709 | 0.2389 | 0.597 | 0.0235 |
| 2021 | hgb | 489 | 0.6671 | 0.2373 | 0.584 | 0.0180 |
| 2021 | market | 489 | 0.6287 | 0.2196 | 0.643 | 0.0277 |
| 2022 | coin_flip | 487 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2022 | elo | 487 | 0.6839 | 0.2453 | 0.577 | 0.0403 |
| 2022 | logreg | 487 | 0.6619 | 0.2351 | 0.602 | 0.0194 |
| 2022 | hgb | 487 | 0.6539 | 0.2312 | 0.612 | 0.0150 |
| 2022 | market | 487 | 0.6024 | 0.2078 | 0.679 | 0.0238 |
| 2023 | coin_flip | 412 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2023 | elo | 412 | 0.6865 | 0.2467 | 0.542 | 0.0075 |
| 2023 | logreg | 412 | 0.6478 | 0.2283 | 0.609 | 0.0340 |
| 2023 | hgb | 412 | 0.6548 | 0.2313 | 0.595 | 0.0294 |
| 2023 | market | 412 | 0.5865 | 0.2004 | 0.698 | 0.0428 |
| 2024 | coin_flip | 380 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2024 | elo | 380 | 0.6836 | 0.2454 | 0.542 | 0.0110 |
| 2024 | logreg | 380 | 0.6606 | 0.2346 | 0.571 | 0.0494 |
| 2024 | hgb | 380 | 0.6464 | 0.2277 | 0.597 | 0.0339 |
| 2024 | market | 380 | 0.5909 | 0.2022 | 0.700 | 0.0322 |
| 2025 | coin_flip | 493 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2025 | elo | 493 | 0.6824 | 0.2446 | 0.561 | 0.0267 |
| 2025 | logreg | 493 | 0.6362 | 0.2226 | 0.641 | 0.0455 |
| 2025 | hgb | 493 | 0.6350 | 0.2219 | 0.677 | 0.0789 |
| 2025 | market | 493 | 0.5871 | 0.2014 | 0.682 | 0.0302 |
| 2026 | coin_flip | 113 | 0.6931 | 0.2500 | 0.500 | 0.0000 |
| 2026 | elo | 113 | 0.6787 | 0.2428 | 0.580 | 0.0460 |
| 2026 | logreg | 113 | 0.5913 | 0.2010 | 0.770 | 0.1552 |
| 2026 | hgb | 113 | 0.6087 | 0.2091 | 0.726 | 0.1136 |
| 2026 | market | 113 | 0.5244 | 0.1737 | 0.757 | 0.0720 |

### Information test: does `hgb` know anything the market does not?

Each quarter a logistic regression with two inputs, logit of the market probability and logit of the `hgb` prediction (no intercept, both orientations, no penalty), is fitted on the out-of-sample predictions of earlier quarters only (fights with odds; 2015 is burn-in), then predicts the quarter. Scored from 2016-01-01 on the same fights as the market alone.

| model | fights | log loss | Brier | accuracy | ECE |
|---|---|---|---|---|---|
| market alone | 4715 | 0.6068 | 0.2098 | 0.665 | 0.0175 |
| market + hgb stack | 4715 | 0.6051 | 0.2091 | 0.673 | 0.0124 |

Log loss, stack minus market alone: -0.0017, 95% interval [-0.0037, +0.0005] (paired bootstrap over events, 1000 draws). Negative means the model adds information the market lacks; an interval that includes zero means no evidence that it does.

Coefficients of the final quarter's fit (2026-01-01): market logit 1.027, hgb logit 0.269.


### Reading

The market beats every model clearly: its log loss is about 0.05 lower than the best model's (`hgb`), with an
interval far from zero, in every year. The information test finds no evidence that `hgb` adds to the market:
the stacked model is at most a few thousandths better, with an interval that includes zero, and the fitted
weights lean on the market (about 1.0) over `hgb` (about 0.3). The odds cover 88.6% of scored fights: blank
odds in the source (mostly 2023-2024) and fights after the source's last date (2026-03-28) have none, so
2026 is only the first quarter. Odds are pre-fight moneylines with the book and timing unknown, so this is a
benchmark, not a statement about closing lines.
