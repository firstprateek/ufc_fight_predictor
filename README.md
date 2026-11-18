# UFC fight predictor

Predicts the winner of UFC fights with calibrated probabilities, judged by log loss on fights the model
has not seen. Milestone 1 (this state of the repo) is the data pipeline: it downloads a public dataset
of ufcstats.com results and turns it into four validated Parquet tables (`events`, `fights`,
`fight_stats_round`, `fighters`). Milestone 2 adds point-in-time features (`ufc features`) and, next, a walk-forward backtest.

```
uv sync
uv run ufc fetch      # download the source CSVs (six small files) into data/raw/
uv run ufc build      # build data/processed/*.parquet and validate them
uv run ufc validate   # re-run validation on the built tables
uv run ufc features --check   # point-in-time features.parquet, plus the leak check (about 2 min)
uv run pytest         # offline tests
```

The data comes from the public `Greco1899/scrape_ufc_stats` dataset (GPL-3.0); it is downloaded at run
time and not stored in this repo.

Design, table schemas and the plan: [`docs/design.md`](docs/design.md). The 2018 class project that
started this repo is kept in [`legacy/2018/`](legacy/2018/) for history only; nothing here uses it.
