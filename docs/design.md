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
  features/         point-in-time features
  backtest/         walk-forward evaluation       (milestone 2, part 2)
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

## Milestone 2: features and backtest

Built in two parts: point-in-time features (below), then the walk-forward backtest and report.

### Features (`src/ufc/features/`, `ufc features [--check]` -> `data/processed/features.parquet`)

One row per fight: `fight_id`, `event_id`, `date`, `fighter_1_id`, `fighter_2_id`, then per-fighter
columns prefixed `f1_` / `f2_` and `diff_` columns (= f1 minus f2). No label is stored; the backtest
joins `fights`. Target (later): `p_f1`, the probability that `fighter_1_id` wins; only `outcome ==
"win"` fights are trained on and scored. Draws and no contests still count as fights in history.

**Point-in-time rule.** Fights are processed one date at a time. Every fight of a date is featurised
from the state before that date, then the state is updated. So a fight sees only fights dated strictly
earlier; same-day earlier bouts (old tournament nights) are ignored on purpose. `fighters` is a current
snapshot: only `height_cm`, `reach_cm`, `stance` and `dob` are used from it. Known limitation: these
are today's values, so a reach or stance changed after a fight is not seen as it was then, and the
truncation guard (which truncates fights and stats, not the snapshot) cannot detect that. Also,
`reach_cm` is null for most fights before about 2010 and almost none after 2018, so its missingness
proxies the era.

Features (each exists as `f1_x`, `f2_x`, `diff_x` unless noted):

- Experience: `n_fights`, `n_wins`, `n_losses`, `win_rate` (wins / prior fights, draws and no contests
  in the denominator), `win_streak`, `loss_streak` (a draw or no contest ends both), `days_since_last`,
  `ko_win_share`, `sub_win_share` (of wins), `ko_loss_share` (of losses). Method mapping: `KO/TKO` and
  `TKO - Doctor's Stoppage` are KO; `Submission` is submission; everything else (decisions, DQ, could not
  continue, overturned, other) is neither.
- Physical: `age` (years at fight date from `dob`), `height_cm`, `reach_cm`, `stance` (strings, no diff)
  and `southpaw` (1.0 / 0.0, null if stance unknown).
- Rates, each as `x_career` (ratio of sums over all prior fights with the stat) and `x_ewm`
  (exponentially weighted, **half-life 3 fights**: the most recent prior fight has weight 1, the one
  before 0.5^(1/3), and so on): `slpm`, `sapm` (significant strikes landed / absorbed per minute),
  `str_acc`, `str_def`, `td15` (takedowns landed per 15 min), `td_acc`, `td_def`, `sub15`, `kd15`,
  `kda15` (knockdowns scored / suffered per 15 min), `ctrl_share`, `ctrled_share` (share of fight time
  in control / being controlled). Duration = (end_round - 1) x 300 + end_time_s seconds (assumes
  5-minute rounds, so the ~180 early fights with 1 or 2 scheduled rounds get inexact per-minute rates). A fight
  without round stats for both fighters, or without an end time, adds nothing to the rates but still
  counts for experience. A rate is null when its denominator is 0 (for example takedown accuracy
  with no attempts).
- Elo: `elo`, one rating per fighter, start 1500, **K = 32**, standard 400-point logistic expectation.
  Updated fight by fight in date order; a win scores 1, a draw 0.5, a no contest does not update.
  Fights on one date all use the pre-date ratings. A debutant has rating 1500 (it is the definition of
  the rating, not a missing value).
- A fighter with no prior fights has `n_fights` 0 and nulls for every other history feature
  (`n_wins`, `n_losses` included), never 0. A diff is null if either side is null.

Leak guard: `ufc features --check` rebuilds the features for a random sample of 200 real fights with
every fight dated on or after them removed from the inputs (the fight itself kept) and requires an
identical row (`--seed` changes the sample); it exits non-zero otherwise. It cannot see a dependence on
the fight's own result or stats (the fight is kept), which has its own unit test. The same property is a unit test on fixtures.

## Later

Skill ratings with uncertainty per dimension; historical and live odds; method and round prediction;
TabPFN in the ensemble; LLM-read news signals (evaluated only on fights after the model's cutoff);
weekly scheduled run.
