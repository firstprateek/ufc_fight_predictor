# UFC fight predictor

Predicts the winner of UFC fights with calibrated probabilities, judged by log loss on fights the model
has not seen. Milestone 1 (this state of the repo) is the data pipeline: it downloads a public dataset
of ufcstats.com results and turns it into four validated Parquet tables (`events`, `fights`,
`fight_stats_round`, `fighters`). Milestone 2 adds point-in-time features (`ufc features`) and a walk-forward backtest of four baselines (`ufc backtest`); results in [`docs/results/baseline.md`](docs/results/baseline.md). Models are also scored against the betting market ([`docs/results/market.md`](docs/results/market.md)). Skill-rating, opponent-adjusted and context features, evaluated under a development / holdout protocol, are in [`docs/results/ratings.md`](docs/results/ratings.md); its numbers use the pinned snapshots `ufc fetch --sha 96adace863c0a793579cc06d27d468797545607f --odds-sha 31d2941509506c9f20e02a5f1de9c253c5289f9c` (plain `ufc fetch` takes the latest upstream, so a later run differs slightly).

## Predict a fight

After `uv sync && uv run ufc fetch && uv run ufc build` (the first time, and whenever you want newer results):

```
uv run ufc predict "Fighter A" "Fighter B" [--date YYYY-MM-DD] [--weight-class NAME] [--rounds 3|5] [--title] [--odds PRICE_A PRICE_B]
uv run ufc predict --card card.csv     # one row per fight; see examples/card.csv
```

The data holds completed fights only, so you name the fight. The probability comes from the same
features and the same `blend` model as the backtest, trained on every decided fight before `--date`
(default: today). Names ignore case and accents; a name shared by two fighters is refused with their
ids (`--id-1`, `--id-2`). `--json` prints the same content as JSON; `--log` appends to the gitignored
`predictions/log.csv` for scoring later. Over the 5,179 past fights with odds the market was the better
estimate (log loss 0.6092 against 0.6474), so the market is the better estimate when the two
disagree. Real output, as the model saw the 2026-11-14 title fight the day before it (`--date`
before a fight ignores that fight and everything after it):

```
$ uv run ufc predict "Ciryl Gane" "Josh Hokit" --date 2026-11-13 --title
Ciryl Gane vs Josh Hokit
2026-11-13  |  Heavyweight  |  title fight, 5 rounds

  Ciryl Gane     66.1%   fair price -195
  Josh Hokit     33.9%   fair price +195
  blend = mean of the logits of hgb and logistic regression; for Ciryl Gane: hgb 63.7%, logistic regression 68.4%

                                    Ciryl Gane          Josh Hokit
  UFC record (W-L-D/NC)                 11-2-1               4-0-0
  Prior UFC fights                          14                   4
  Last fight (days since)   2026-06-14 (152 d)  2026-06-14 (152 d)
  Age                                     36.6                29.0
  Reach (cm)                               206                 185
  Elo                                     1653                1570
  Glicko                                  2501                2100

Data snapshot 96adace863c0a793579cc06d27d468797545607f; latest fight in the data 2026-11-14; trained on 7,671 fights.
```

```
uv sync
uv run ufc fetch      # download the source CSVs (six small files) and the odds file into data/raw/
uv run ufc build      # build data/processed/*.parquet (including odds.parquet) and validate them
uv run ufc validate   # re-run validation on the built tables
uv run ufc features --check   # point-in-time features.parquet, plus the leak check (about 2 min)
uv run ufc tune-ratings --before 2015-01-01   # re-derive the committed rating constants from pre-2015 fights
uv run ufc backtest   # walk-forward backtest incl. the market benchmark; report in reports/, summaries in docs/results/baseline.md and docs/results/market.md
uv run ufc backtest --end 2022-12-31 --features-set base   # development period only; --ablation runs every feature set there
uv run ufc ratings-report --frozen-commit <sha>   # docs/results/ratings.md: ablation, final vs base on development and holdout, market (about 4 min)
uv run pytest         # offline tests
```

The data comes from the public `Greco1899/scrape_ufc_stats` dataset (GPL-3.0); it is downloaded at run
time and not stored in this repo. Historical betting odds (the benchmark the models are measured against) come from
`shortlikeafox/ultimate_ufc_dataset` (Apache-2.0), also downloaded at run time; only five of its columns are read.

Design, table schemas and the plan: [`docs/design.md`](docs/design.md). The 2018 class project that
started this repo is kept in [`legacy/2018/`](legacy/2018/) for history only; nothing here uses it.
