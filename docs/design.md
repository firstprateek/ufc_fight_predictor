# UFC fight predictor v2: design

Source of truth for what we are building and why. Keep it short; update it when a decision changes.

## Goal

For any scheduled UFC fight, output a calibrated probability that each fighter wins (later: method and
round too). Quality is judged by log loss and calibration on fights the model has never seen, compared
with the betting market's closing odds. Winner accuracy is reported but is not the target.

The 2018 class project is kept under `legacy/2018/` for history only. Nothing in v2 depends on it.

## Principles

1. **Point in time.** Every feature for a fight uses only information available before that fight.
   Any code that could leak a result into an input is a bug, however small.
2. **Walk forward.** Models are tested by training on the past and predicting the next events, in date
   order. Never random splits.
3. **No order leaks.** The stats site lists the winner first on event pages. Stored fights never encode
   the result in fighter order: `fighter_1_id < fighter_2_id` (string sort), with `winner_id` separate.
4. **Raw first.** Every fetched file is kept on disk untouched. Parsing reads that copy, so parsers can
   be fixed and re-run without re-fetching.
5. **Small and boring.** About 8,000 fights. Plain Python, well-known libraries, runs on one Mac.

## Layout

```
src/ufc/            package (CLI entry point: `ufc`)
  fetch.py          download source CSVs into the raw cache
  parse.py          raw cells -> typed values (pure functions)
  build.py          source CSVs -> validated Parquet tables
  validate.py       table validation
  features/         point-in-time features        (milestone 2)
  backtest/         walk-forward evaluation       (milestone 2)
overrides/          committed manual fixes for the build (fighter ids, dropped fights)
tests/              pytest, offline, uses small hand-made CSV fixtures
data/raw/           source snapshots              (gitignored)
data/processed/     Parquet tables                (gitignored)
docs/               this file, decisions
legacy/2018/        the old project
```

Tooling: `uv`, Python 3.12, `ruff`, `pytest`. Data files are never committed.

## Milestone 1: data pipeline

Source: the public dataset `Greco1899/scrape_ufc_stats` on GitHub (six CSV files of ufcstats.com data,
refreshed weekly, GPL-3.0). We download the CSVs at run time and never commit them or copy that repo's
code. We do not scrape ufcstats.com ourselves: it sits behind a JavaScript bot check (found 2026-11-17),
and we don't work around it.

Fetching: `ufc fetch` downloads the six CSVs at one upstream commit into `data/raw/<commit-sha>/` and
records that sha, so every build is reproducible from a named snapshot. Generic User-Agent, no personal
details.

Known shape of the source, which the build must handle:

- Fight, event and fighter ids come from the `URL` columns (the hex id at the end).
- `ufc_fight_stats.csv` has no URL: join it to fights on (event name, bout), trimming whitespace.
- Fights name their fighters but give no fighter ids. Resolve names to ids through the fighter tables.
  A few names belong to more than one fighter: pick the candidate that is plausible for the fight (age
  at the fight date, listed weight within 15 lb of the division limit) when only one is. Otherwise use
  the committed `overrides/fighter_ids.csv` (name -> id, with a note; it also fixes names the fight
  pages spell differently from the fighter pages). A fight that still cannot be resolved fails the
  build unless it is listed in `overrides/dropped_fights.csv` with a reason; we never guess.
- `OUTCOME` is `W/L`, `L/W`, `D/D` or `NC/NC`, in the order the bout names the fighters.
- Cells look like `41 of 65`, `0:03`, `5' 11"`, `--` and `---`; `--` and `---` mean null.

Tables in `data/processed/`:

| Table | One row per | Key columns |
|---|---|---|
| `events` | event | `event_id`, `name`, `date`, `location` |
| `fights` | fight | `fight_id`, `event_id`, `date`, `bout_order`, `fighter_1_id`, `fighter_2_id`, `winner_id` (null for draw / no contest), `outcome` (win, draw, nc), `method`, `end_round`, `end_time_s`, `scheduled_rounds`, `weight_class`, `is_title`, `gender`, `referee` |
| `fight_stats_round` | fight, fighter, round | knockdowns; significant and total strikes landed / attempted; takedowns landed / attempted; submission attempts; reversals; control seconds; significant strikes by target (head, body, leg) and position (distance, clinch, ground), landed / attempted |
| `fighters` | fighter | `fighter_id`, `name`, `height_cm`, `reach_cm`, `stance`, `dob` |

`bout_order` is 1 for the main event (source order, top of the card first). `end_time_s` is seconds
into `end_round`. `gender` is `male` or `female`; `weight_class` is the division name without "Title
Bout" (`is_title` carries that). Rows the build drops (the source lists some fights twice under a
renamed event, and some events are missing from its events list) are counted and named in the build
output; the `fights` row count must equal the source count minus those.

IDs are the site's own hex ids from its URLs. Units are metric; times are seconds; dates are ISO.
Missing values are null, never 0 or a sentinel.

CLI:

- `ufc fetch` downloads the latest snapshot of the source CSVs.
- `ufc build` parses the newest snapshot into the tables and runs validation.
- `ufc validate` runs validation alone.

Validation (fails loudly): unique ids; `fights` row count equals source fights minus explicit drops; every fight has two distinct fighters that exist in `fighters`;
`winner_id` is one of the two or null; round stats exist for both fighters for rounds 1..`end_round`
(older fights may lack round stats: report the count, don't fail); landed <= attempted; dates parse.

## Milestone 2: features and backtest (not started)

Point-in-time feature builder, walk-forward backtest, baselines (coin flip, Elo, logistic regression on
rating and physical differences, gradient-boosted trees), metrics (log loss, Brier, accuracy,
calibration), one report per run.

## Later

Skill ratings with uncertainty per dimension; historical and live odds; method and round prediction;
TabPFN in the ensemble; LLM-read news signals (evaluated only on fights after the model's cutoff);
weekly scheduled run.
