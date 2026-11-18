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
6. **Snapshot fields carry survivorship.** `fighters` is today's snapshot. Whether a fighter's reach,
   stance, height or date of birth is filled in depends on how long they lasted in the UFC, so its
   missingness reveals the future. It must never be a model input: no missing flags, no nulls the
   model can branch on. The physical fields are imputed with static rules that use no fight results
   (see Features), and `ufc features` fails if any physical column holds a null. Imputing is not
   enough: an imputed value is a recognisable constant, and a model can learn "this exact value
   means the fighter did not last". So a model never sees a label next to a filled-in value: fights
   where either fighter has any filled-in field (`snapshot_imputed`) are left out of training (the
   flag itself is never a model input) but are still predicted and scored. Evidence (found in
   milestone 2, measured on decided fights where exactly one fighter has the value; the share is how
   often the fighter with the value won):

   | Field | 2010-2014 | 2015-2016 | 2017-2019 | 2020-2026 |
   |---|---|---|---|---|
   | reach | 261 of 1,748 fights, 93% | 120 of 947, 98% | 23 of 1,421, 91% | 93 of 3,477, 46% |
   | stance | 24, 88% | 19, 100% | 1, 100% | 83, 46% |
   | age | 8, 88% | 0 | 0 | 10, 60% |
   | height | 3, 100% | 0 | 0 | 94, 46% |

   The first baseline exploited this (the tree model's log loss was 0.590 in 2015 and drifted to about
   0.65 later) and the truncation check cannot see it, because the snapshot is not truncated.

7. **Odds are read through five columns.** The odds file also holds the winner and many
   result-derived columns. `odds.load_odds_source` reads only `R_fighter`, `B_fighter`, `R_odds`,
   `B_odds` and `date` (pandas `usecols`), and a test enforces it. Nothing else from that file may
   be loaded, however convenient: a result column next to the prices is a leak waiting to happen.
8. **Choices are made on development data only.** The test period is split: **development** is
   2015-01-01 to 2022-12-31, **holdout** is 2023-01-01 to the latest fight. Every choice (which
   feature groups to keep, anything else) is made on development numbers. The holdout is run once for a
   new configuration, after that configuration is frozen in a commit; the results file names the
   commit. Rating hyperparameters are tuned only on fights dated before 2015-01-01 and never on the
   test period. The rule for the final configuration is written before the numbers exist (see
   "Evaluation protocol" below), and the protocol is not adjusted after seeing results. Code backs this
   up: `ufc backtest --end` drops every later fight before anything is fitted, and
   `backtest/protocol.py` refuses to run a feature set that is not the frozen one (or `base`) past
   2022-12-31.

## Layout

```
src/ufc/            package (CLI entry point: `ufc`)
  fetch.py          download source CSVs into the raw cache
  parse.py          raw cells -> typed values (pure functions)
  build.py          source CSVs -> validated Parquet tables
  validate.py       table validation
  odds.py           historical odds: five-column loader, matching to fights, odds.parquet
  features/         point-in-time features
  backtest/         walk-forward evaluation
overrides/          committed manual fixes for the build (fighter ids, dropped fights, odds names)
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

Built in two parts: point-in-time features, then the walk-forward backtest and report (both below).

### Features (`src/ufc/features/`, `ufc features [--check]` -> `data/processed/features.parquet`)

One row per fight: `fight_id`, `event_id`, `date`, `fighter_1_id`, `fighter_2_id`, then per-fighter
columns prefixed `f1_` / `f2_` and `diff_` columns (= f1 minus f2). No label is stored; the backtest
joins `fights`. Target (later): `p_f1`, the probability that `fighter_1_id` wins; only `outcome ==
"win"` fights are trained on and scored. Draws and no contests still count as fights in history.

**Point-in-time rule.** Fights are processed one date at a time. Every fight of a date is featurised
from the state before that date, then the state is updated. So a fight sees only fights dated strictly
earlier; same-day earlier bouts (old tournament nights) are ignored on purpose. `fighters` is a current
snapshot: only `height_cm`, `reach_cm`, `stance` and `dob` are used from it, and never their
missingness (principle 6). Known limitation: these are today's values, so a reach or stance changed
after a fight is not seen as it was then, and the truncation guard (which truncates fights and stats,
not the snapshot) cannot detect that.

Features (each exists as `f1_x`, `f2_x`, `diff_x` unless noted):

- Experience: `n_fights`, `n_wins`, `n_losses`, `win_rate` (wins / prior fights, draws and no contests
  in the denominator), `win_streak`, `loss_streak` (a draw or no contest ends both), `days_since_last`,
  `ko_win_share`, `sub_win_share` (of wins), `ko_loss_share` (of losses). Method mapping: `KO/TKO` and
  `TKO - Doctor's Stoppage` are KO; `Submission` is submission; everything else (decisions, DQ, could not
  continue, overturned, other) is neither.
- Physical: `age` (years at fight date from `dob`), `height_cm`, `reach_cm`, `stance` (strings, no diff)
  and `southpaw` (1.0 / 0.0). These are never null: a missing snapshot value is imputed by a static
  rule fitted once on the full tables, using fight dates and divisions but no results. height: median
  height of fighters in the same weight class and gender (a fighter's class and gender are those of
  their most frequent division; fall back to the gender median, then the global median). reach: the
  fighter's (possibly imputed) height times the global median reach-to-height ratio. stance:
  `Orthodox`. age: the global median age at fight time. Diffs are computed after imputation.
  `ufc features` exits non-zero if any physical column (per fighter or diff) contains a null. The
  leak check reuses the constants fitted on the full tables. The constants and each fighter's
  division use fight dates and divisions of all fights, including later ones (no results), which
  the truncation check cannot see. Each row also has `f1_snapshot_imputed` and `f2_snapshot_imputed`
  (boolean; true when any of that fighter's height, reach, stance or dob was filled in). They are
  bookkeeping for the backtest and never model inputs (`feature_columns` excludes them). An imputed
  value is a constant, so on its own it still marks the fighter who did not last; the backtest
  therefore keeps those fights out of training (below).
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

### Feature groups (`ratings`, `adjusted`, `context`)

Added on top of the features above; all point in time (same-date rule: every fight of a date is
featurised from the state before the date, updates apply after it) and each exists as `f1_`, `f2_`
and `diff_` columns. A debutant's rating is the defined prior (like Elo's 1500), never a null. The
tuned constants live in `features/params.py`; `ufc tune-ratings --before 2015-01-01` re-derives them
(see "Tuning" below).

**`ratings`**

- Glicko-1 (`glicko_r`, `glicko_rd`, and the pair feature `glicko_exp`, the expected score). Each date
  is a rating period. A debutant has rating 1500 and deviation 350. Between fights the deviation grows
  as `rd = min(350, sqrt(rd_after_last_fight^2 + c^2 x days))`; `c` is tuned. `glicko_exp` uses
  Glicko's `g` of the root-sum-square of both deviations. A no contest changes nothing and does not
  restart the clock.
- Method-weighted Elo (`welo`): like `elo`, but each update is `K x m(method) x (score - expected)`.
  `m` is 1 for split and majority decisions and tuned for unanimous decisions, KO/TKO (with doctor's
  stoppage), submissions and everything else (DQ, overturned, could not continue, other); `K` is tuned.
  A draw uses its method's multiplier too; a no contest moves nothing. The multiplier grid starts at 0,
  and the tuned value for the "other" endings is 0 (they do not move the rating). Plain `elo` stays
  (K = 32).

**`adjusted`** (opponent-adjusted performance). Five dimensions, each measured per fight from round
stats as an output of the fighter: `slpm` (significant strikes landed per minute), `td15` (takedowns
landed per 15 minutes), `ctrl_share` (share of fight time in control), `kd15` (knockdowns per 15
minutes) and `sub15` (submission attempts per 15 minutes). Per dimension a fighter has an offence and a
defence rating in standard-deviation units, starting at 0. The model of A's value against B is
`league mean + sd x (offence_A - defence_B) + noise` (a high defence suppresses the opponent). After each
date, each observation with error `e = (value - mean) / sd - (offence_A - defence_B)` moves
`offence_A += rate_A x e` and `defence_B -= rate_B x e`, where
`rate = max(rate_min, 1 / (fights_with_stats + k0))` per fighter. Mean and sd are the running mean and
standard deviation of every observation before the date; until a dimension has 30 observations
nothing is updated (the sd would mean nothing); the per-fighter fight count `fights_with_stats` still
 counts those early fights. Fights without round stats update nothing. Columns:
`adj_<dim>_off`, `adj_<dim>_def` and the pair feature `adj_<dim>_exp` (the expected value in the
dimension's own units, `mean + sd x (own offence - opponent defence)`); `diff_adj_<dim>_exp` is the
expected differential.

**`context`** (all `ctx_*`). Strength of schedule from the plain Elo each opponent had before that
fight: `sos_mean` (prior opponents), `sos_win_mean` and `best_win_elo` (opponents beaten), `last_opp_elo`,
`last_result` (+1 win, -1 loss, 0 draw or no contest), `last_finish` (1 if the last fight ended in a
KO/TKO or submission), `elo_change3` (Elo now minus Elo before the last three fights, or fewer if
fewer), `elo_peak_gap` (highest Elo after any fight, at least 1500, minus now). Activity and
setting: `fights_24m` (prior fights in the last 730 days), `wc_fights` (prior fights in this weight
class), `wc_first` (1 if none), `wc_move` (-1, 0 or +1: this division's weight limit against the last
fight in a ranked division; null for catch weight, open weight, tournaments or no prior fight),
`five_round_fights`, `title_fights` (prior), and for this fight `is_five_rounds` and `is_title`
(announced before the fight; equal for both fighters, so their diffs are 0, except that
`is_five_rounds` is null for the 31 fights with no scheduled-rounds value). History features are
null for debutants; counts are 0.

The announced facts of the fight itself (division, scheduled rounds, title) reach the trackers
through a `Card`; the outcome and the fight's own stats only reach them after the date.

### Tuning (`ufc tune-ratings --before 2015-01-01`)

Rating hyperparameters are tuned only on fights dated before 2015-01-01 (the start of the test period).
Each candidate is scored by its one-step-ahead error on fights from 2005-01-01 to the cutoff (every
fight is predicted from the state before its date, then the state updates; earlier fights warm the
ratings up). Glicko `c`: grid search on log loss of the expected score. Method-weighted Elo: coordinate
descent over `K` and the four multipliers on log loss. Adjusted model: grid search over `k0` and
`rate_min` on the mean over the five dimensions of the standardised squared error. The grids are in
`features/tune.py`; the search is deterministic. The command refuses a cutoff later than 2015-01-01,
fights dated on or after the cutoff never reach the tuner (tested by corrupting them), and it exits
non-zero if the result differs from the committed constants.

Leak guard: `ufc features --check` rebuilds the features for a random sample of 200 real fights with
every fight dated on or after them removed from the inputs (the fight itself kept) and requires an
identical row (`--seed` changes the sample); it exits non-zero otherwise. It cannot see a dependence on
the fight's own result or stats (the fight is kept), which has its own unit test. The same property is a unit test on fixtures.
### Backtest (`src/ufc/backtest/`, `ufc backtest [--start 2015-01-01]`)

Only `outcome == "win"` fights are scored; `y` = 1 if `fighter_1_id` won. The test period is every
scored fight from the start date to the latest, split into calendar quarters. For each quarter every
model is fitted on all scored fights dated before the quarter starts (including all history before the
start date) and predicts the quarter. Elo and features are already point-in-time and are not refit.

Models (hyperparameters fixed in `backtest/models.py`, never tuned on the test period):

1. `coin_flip`: 0.5.
2. `elo`: Elo expected score from the pre-fight ratings (K = 32, see above).
3. `logreg`: logistic regression (C = 1) on `diff_elo`, `diff_age`, `diff_reach_cm`,
   `diff_n_fights`, `diff_win_rate` and the striking differential (`diff_slpm_career - diff_sapm_career`);
   train-set median imputation with missing flags, then standard scaling. The physical inputs are never
   null, so no flag can encode snapshot missingness; flags remain only for history features (debutants).
   Fit only on fights without a `snapshot_imputed` flag (see below).
4. `hgb`: scikit-learn `HistGradientBoostingClassifier` (200 iterations, learning rate 0.05, max depth
   3, min 40 samples per leaf, L2 1.0, no early stopping, seed 0) on every numeric `f1_`/`f2_`/`diff_`
   column except the `snapshot_imputed` flags; nulls are handled natively. Columns with fewer than two
   distinct training values are dropped. Also fit only on unflagged fights.

Snapshot exclusion: a fight where either fighter has a `snapshot_imputed` flag is never in a
training set (`logreg` and `hgb`; Elo is untouched), in either orientation, and the flags are never in
a feature matrix; the run fails (`LeakSuspected`) if either happens. Flagged fights in the test period
are still predicted and scored, with the neutral imputed values, and the report shows how many fights
are excluded from training (overall and in the last fold) and the scores on flagged versus unflagged
scored fights.

Symmetry: fitted models train on each fight in both orientations (`swap_orientation`: f1/f2 swapped,
label flipped, diffs negated) and predict `p = (p(f1,f2) + 1 - p(f2,f1)) / 2`.

Report (`reports/<snapshot-sha>-<date>/report.md` and `predictions.parquet`, gitignored; the
summary is committed as `docs/results/baseline.md`): per model log loss, Brier, accuracy and ECE overall
and per year; a calibration table for the best model; a paired bootstrap (events resampled, 1,000
draws, seed 0, 95% percentile interval) of each model's log-loss difference from Elo. ECE uses 10
equal-width bins with every fight counted from both fighters' views; accuracy counts p = 0.5 as half a hit.

Leak guards: the truncation check above; the orientation check (each quarter, `logreg` and `hgb`
must give `1 - p` for swapped fighters, and training dates must precede the quarter, else the run
fails); and a tripwire: if any model's overall accuracy exceeds 75% the report is written, the command
exits non-zero and says a leak is likely. Real accuracy is expected around 55-65%. The tripwire only
catches gross leaks (a subtle one would sit under 75%); the real protection is the point-in-time
construction, the truncation check (a 200-fight sample) and the tests that flip later labels and results.

### Evaluation protocol (`ufc backtest --end`, `--features-set`, `--ablation`)

- `ufc backtest --end YYYY-MM-DD` ends the test period at that date. Fights after it are dropped before
  fitting, so no later label or feature row can influence a result (tested by changing both). The feature table
  itself is outside that guarantee: the imputation constants are fitted on all fights, results excluded
  (see Features).
- `ufc backtest --features-set <set>` runs a feature set: `base` (the features above), `ratings`,
  `adjusted`, `context` (each is `base` plus that group), `all`, or a join such as `ratings+adjusted`.
  A column belongs to a group by name (`glicko_*` and `welo`; `adj_*`; `ctx_*`). For a set, `hgb` uses
  every numeric column of the set; logistic regression uses its base inputs plus `diff_glicko_r` and
  `diff_welo` (ratings) and the five `diff_adj_<dimension>_exp` expected differentials (adjusted).
  Hyperparameters of `hgb` and logistic regression do not change. The `context` group adds no
  logistic-regression input, so its `logreg` rows equal `base` and its `blend` rows differ from `base`
  only through `hgb`.
- `blend` is the mean of the logits of `hgb` and logistic regression (a prediction-level average, no
  extra fit); it has its own row in the tables and its own information test.
- `ufc backtest --ablation` runs `base`, `ratings`, `adjusted`, `context` and `all` on the development
  period and applies the pre-registered rule: the final configuration is `all`, minus any group whose
  `hgb` development log loss, as the set `base` plus that group, is worse than `base` with a
  paired-bootstrap 95% interval (events resampled, 1,000 draws) that excludes zero. No other
  selection. (The rule compares each group with `base` rather than `all` with `all` minus the group:
  that is what the ablation table shows, fixed before any number existed.)
- `ufc ratings-report --frozen-commit <sha>` writes `docs/results/ratings.md`: it checks that the commit
  is an ancestor of HEAD and holds the frozen configuration, re-runs the development ablation and
  requires the decision rule to give the frozen configuration, then runs the frozen configuration and
  `base` on development, holdout and the full period, against the market, with the constants.
- Until a commit freezes the configuration (`FINAL_FEATURE_SET` in `backtest/protocol.py`), only `base`
  may run past `--end 2022-12-31`; afterwards only `base` and the frozen set may. A feature set
  whose columns are not in `features.parquet` does not run.

## Milestone 3: market odds

The betting market is the benchmark the models are measured against. It has two parts: the odds table
(below) and the benchmark in the backtest ("Against the market", after it).

Source: GitHub `shortlikeafox/ultimate_ufc_dataset` (Apache-2.0), file `ufc-master.csv` at the repo root
(7,177 fights from 2010-03-21 to 2026-03-28 at the pinned commit). Odds are American moneylines in
`R_odds` and `B_odds`; the bookmaker and how close to the fight they were taken are not
documented, so we call them "pre-fight moneyline, book and timing unknown". `ufc fetch` downloads the
file at the head of `main` (or at `--odds-sha`) into `data/raw/odds/<sha>/` and records the sha;
it is never committed. Only the five columns in principle 7 are read. Rows with a blank price on
either side are skipped (about 250, mostly 2023-2024).

`ufc build` then writes `data/processed/odds.parquet`, one row per matched fight: `fight_id`,
`odds_f1`, `odds_f2` (American, in our `fighter_1` / `fighter_2` orientation), `p_market_f1` (each side's
implied probability, `100 / (o + 100)` for a positive price and `-o / (-o + 100)` for a negative one,
normalised so the two sum to 1, which removes the bookmaker margin) and `overround` (the sum before
normalising). Without a fetched odds snapshot the build still succeeds and says odds were skipped.

Matching an odds row to a fight:

1. Names are normalised: case, accents, punctuation (hyphens count as spaces), spacing and the suffixes
   Jr, Sr, II, III, IV. Our names come from `fighters.name` via the fight's fighter ids.
2. Main pass: the same unordered pair of normalised names, with the odds date within one day of the
   fight date.
3. Fallback pass, same date only: both fighters' surnames equal ours (the two surnames differ from each
   other) and exactly one fighter's full normalised name is identical (for example `Nick` against
   `Nicholas`).
4. Manual aliases in the committed `overrides/odds_names.csv` (`odds_name`, `fighter_name`, `note`),
   applied before the passes. Each row is a fighter the odds file names differently from ufcstats
   (a later name change, a nickname, a spelling), with the evidence in the note.
5. Never guess. An odds row that fits no fight, or more than one, is left unmatched and listed in the
   build output. A fight claimed by two odds rows is left unmatched for both.

Validation: `p_market_f1` strictly between 0 and 1; overround between 0.95 and 1.25 (rows outside are
dropped, counted and listed: they are corrupt prices, for example two long underdogs); one odds row per
fight; every `fight_id` exists in `fights`. `ufc validate` re-checks the built table. The build prints
coverage per year (source rows with odds, matched, unmatched) and the share of scored fights (decided,
from 2015-01-01) that have odds.

### Against the market (`ufc backtest`, `docs/results/market.md`)

`market` is a fifth model whose prediction is `p_market_f1`; it needs no training and has no row in the
all-fights tables, which are unchanged (`MODELS` does not contain it, and no other model ever sees the odds,
so odds are not a feature). The backtest loads `odds.parquet` if it exists and adds an "Against the market"
section scored only on test-period fights that have odds, the same fights for every model: overall and
per-year log loss, Brier, accuracy and ECE, a calibration table for the market, and a paired bootstrap
(events resampled, 1,000 draws, 95%) of each model's log loss minus the market's. Expect the market to win.

The market favourite should win roughly 65% of the time. The 75% accuracy tripwire applies to `market`
too (on the fights with odds); a market that looks much better or much worse than 65% is first suspected
of an orientation or matching bug.

**Information test.** It answers "does our model know anything the market does not?". Each quarter from
2016 (2015 is burn-in), a logistic regression on two inputs, the logit of `p_market_f1` and the logit of the
`hgb` prediction, is fitted on the out-of-sample predictions of earlier quarters only (fights with odds,
both orientations, no intercept, no penalty), and predicts the quarter. Its log loss is compared with the
market alone on the same fights (paired bootstrap), and the final quarter's coefficients are reported.
A stack that beats the market with an interval below zero would mean `hgb` carries information the market
lacks; an interval including zero means no evidence of that. This is a diagnostic, not a model for
betting: no return-on-investment simulation is done.

## Later

Skill ratings with uncertainty per dimension; live odds for upcoming fights; method and round prediction;
TabPFN in the ensemble; LLM-read news signals (evaluated only on fights after the model's cutoff);
weekly scheduled run.
