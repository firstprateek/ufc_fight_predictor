# UFC fight predictor

Predicts the winner of UFC fights with calibrated probabilities, judged by log loss on fights the model
has not seen. Milestone 1 (this state of the repo) is the data pipeline: it downloads a public dataset
of ufcstats.com results and turns it into four validated Parquet tables (`events`, `fights`,
`fight_stats_round`, `fighters`). Milestone 2 adds point-in-time features (`ufc features`) and a walk-forward backtest of four baselines (`ufc backtest`); results in [`docs/results/baseline.md`](docs/results/baseline.md). Models are also scored against the betting market ([`docs/results/market.md`](docs/results/market.md)).

```
uv sync
uv run ufc fetch      # download the source CSVs (six small files) and the odds file into data/raw/
uv run ufc build      # build data/processed/*.parquet (including odds.parquet) and validate them
uv run ufc validate   # re-run validation on the built tables
uv run ufc features --check   # point-in-time features.parquet, plus the leak check (about 2 min)
uv run ufc tune-ratings --before 2015-01-01   # re-derive the committed rating constants from pre-2015 fights
uv run ufc backtest   # walk-forward backtest incl. the market benchmark; report in reports/, summaries in docs/results/baseline.md and docs/results/market.md
uv run ufc backtest --end 2022-12-31 --features-set base   # development period only; --ablation runs every feature set there
uv run pytest         # offline tests
```

The data comes from the public `Greco1899/scrape_ufc_stats` dataset (GPL-3.0); it is downloaded at run
time and not stored in this repo. Historical betting odds (the benchmark the models are measured against) come from
`shortlikeafox/ultimate_ufc_dataset` (Apache-2.0), also downloaded at run time; only five of its columns are read.

Design, table schemas and the plan: [`docs/design.md`](docs/design.md). The 2018 class project that
started this repo is kept in [`legacy/2018/`](legacy/2018/) for history only; nothing here uses it.
