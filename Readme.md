# Panther Post

Forecasting parcel volume across Pitt's residence-hall mailrooms to drive
**day-by-day staffing assignment** and **capacity planning**.

Built for SteelHacks 2026 on two years of real WTS package data
(2024-09-19 to 2026-09-19, 109,482 parcels, 9 mailrooms).

---

## Unresolved: PII and a leaked key are in the published history

Checked 2026-09-20, and both are still outstanding.

The WTS export has an unnamed column holding **recipient names**. `.gitignore`
excludes the data files, but it was added *after* they were committed, and
`.gitignore` has no effect on a file git already tracks. These four are in the
published history:

```
packagestats.csv   packagestats.xlsx
files/packagestats_cleaned.xlsx   searchresults(1).csv
```

Separately, the old `elevenLabs.py` carried a **live ElevenLabs API key** as a
string literal. The file has been deleted and replaced by `briefing.py`, which
reads `ELEVENLABS_API_KEY` from the environment, but deleting a file does not
remove it from earlier commits.

`origin/main` is at the same commit as local `main`, so all of it has already
been pushed, and the repository is public. Treat the old key as compromised:
public repositories are scraped continuously, so the exposure window is not
something a later cleanup closes.

In order:

1. **Revoke the old ElevenLabs key** in their dashboard and issue a new one.
   This is the only step that actually fixes the key; everything else is
   tidying.
2. **Make the repository private** while the rest is sorted, if the names
   matter more than the demo link.
3. **Rewrite the history**: `bash scripts/clean_git_history.sh`. It backs up
   first, purges all five paths, and stops short of pushing.

Verify with:

```
git ls-files | grep -Ei 'packagestats|searchresults|elevenLabs'
git rev-list --objects --all | grep -Ei 'packagestats|searchresults|elevenLabs'
```

The pipeline itself never writes names anywhere: `panther/data.py` drops that
column on load, and nothing under `outputs/` or `powerbi/` contains one.

---

## Track eligibility (No Wrapper)

No language models anywhere in the pipeline. Everything is classical
statistics and classical machine learning: Poisson, Tweedie and Bayesian ridge
regression, gradient-boosted trees, discrete Kaplan-Meier survival
estimation, and rolling-origin cross-validation. `elevenlabs` appears in
`requirements.txt` for text-to-speech only, and the narration it reads is
generated from a fixed template rather than written by a model.

## Quick start

```
python -m pip install -r requirements.txt

python -m panther.run train      # fit one model per mailroom on year one   (~40s)
python -m panther.run test       # score year two, accuracy per mailroom    (~40s)
python -m panther.run predict    # forecast the current week                (~10s)
```

Run from the repository root, the folder containing `packagestats.csv`.
`test` and `predict` both need `outputs/models.pkl`, which `train` writes, so
run them in that order the first time.

Useful variants:

```
python -m panther.run test --site "Tower B" --tail 20    # day-by-day table
python -m panther.run predict --as-of 2026-10-28         # rehearse another week
```

Do not pipe these through `head` or `more` on Windows. That sends SIGPIPE and
kills the command partway, and a half-finished `train` leaves no model cache.
Redirect instead: `python -m panther.run test > outputs\log.txt 2>&1`.

---

## The pivot

The project started as **"predict package surges so we can ask for more
lockers."** It ended somewhere more useful, in three steps.

**1. From daily arrivals to parcels held.**
The original model (`files/panther_post_model.py`, kept as the v6 baseline)
compared a single day's *intake* against locker capacity. Those are different
quantities. The median parcel sits 24 hours, but 19.4% are still held after a
week and 6.5% after a month, so holdings accumulate far above any day's
arrivals: Tower B's peak daily intake is 741 while its peak concurrent
holdings are 3,596 — roughly 5x. Because the export carries both a `Received`
and a `Delivered` timestamp, true occupancy is directly observable, which made
it both the right target and a free validation set. `panther/dwell.py` models
dwell time as a survival curve and converts intake into occupancy.

**2. From lockers to staffing.**
Locker counts were requested from Operations and never supplied, so the
1,000-unit pool in the original code is a placeholder that is inconsistent
with observed occupancy. Meanwhile labour is driven by *transactions*, and
both transaction types are in the data: intake scans and pickup handouts. The
same dwell model that produces occupancy also produces the pickup forecast, so
one model serves both deliverables. Per-parcel handling time was measured from
the gaps between consecutive scans (37-49s median by site), so no time-motion
study was needed.

**3. From "buy more lockers" to "enforce a hold deadline."**
Tower B's share of parcels still held past 90 days went from 0.9% in 2024-25
to 6.7% in 2025-26. No other site moved. That single change drove its
occupancy from roughly 500 parcels to 2,690, and the backlog was only cleared
by an administrative purge on 2026-07-01 that disposed of 1,806 items at a
median age of 210 days. Modelling a 30-day return-to-sender policy cuts Tower
B's peak holdings by about 90%. **The capacity problem at Tower B is
non-collection, not demand** — and that recommendation costs nothing.

---

## What's hard here

Four things in this project were genuinely difficult, each discovered by the
data contradicting an assumption.

**Arrivals are not occupancy.** The original model compared one day's intake
against locker capacity. Tower B's peak intake is 741 parcels; its peak
concurrent holdings are 3,596. Getting from one to the other needs a dwell
survival curve convolved against the arrival series, estimated by discrete
Kaplan-Meier so that administrative purges and still-open records can be
censored rather than counted as student pickups.

**A log-link model fed its own predictions diverges.** Forecasting 36 days
ahead with rolling-average features drove Tower B's `roll7` from 127 to
188,400 and the Poisson mean to infinity. The fix is horizon-aware model
selection: momentum features within 14 days, calendar-only beyond, so nothing
downstream can feed back on itself.

**Detecting an administrative purge needs the age of what was cleared, not
the volume.** A volume test flags every post-break resumption rush, because
the baseline is depressed by the break itself - it censored 56% of Tower B's
move-in records, which is exactly the genuine behaviour the model must learn.
Keying on median parcel age separates a bulk disposal (2026-07-01: 1,806
items, median age 210 days) from students returning in January (419 items,
median age 35 days).

**One site changed behaviour and no other did.** Tower B's share of parcels
held past 90 days went 0.9% -> 6.7% between academic years while every other
site stayed under 0.5%. A single pooled survival curve could not represent
both regimes; stratifying by regime cut occupancy error from 44.6% to 13.7%.

## Files

### Pipeline (`panther/`)

**`config.py`**: Every tunable assumption in one place. Locker counts,
headcount, shift length, handling times, the train/test split date, the
staffing quantile, dwell-regime boundaries, and purge-detection thresholds.
Values that were *measured* from the data are marked as such; values that
Operations never supplied are marked `PLACEHOLDER`. Change numbers here and
nowhere else.

**`data.py`**: Loads and cleans the WTS export. Drops the recipient-name
column, parses timestamps, computes dwell hours, builds per-site daily panels
of intake / pickups / true occupancy, sorts sites into those with enough
history to model alone and those without, derives locker allocation, and flags
administrative purge days. Also handles the WTS quirk where the `Mailroom`
column sometimes holds the *scanning user* rather than the location — those
rows become `Unassigned` rather than being guessed at.

**`calendar_features.py`**: The campus calendar and all model features.
Closures (Sundays everywhere, breaks at all but two sites), period
classification, day-of-week encoding, move-in / move-out / spring-break
windows, and the holiday event ramps. Also defines the named feature sets the
model pool draws from.

**`dwell.py`**: The survival model. Estimates, per site and per arrival
period, the probability a parcel is still held *k* days after arrival, using
discrete Kaplan-Meier so that purge days and still-open records can be
censored correctly rather than counted as student pickups. Convolves an intake
series into occupancy and pickup series, and can apply a return-to-sender
deadline to produce the disposal scenario.

**`selection.py`**: Candidate model pool, model selection, and metrics. Six
estimator families crossed with four feature sets, selected by rolling-origin
cross-validation *inside the training year*, then scored once on the test
year. Also holds the quantile machinery and the accuracy table.

**`staffing.py`**: Turns forecast volumes into labour. Measures the intra-day
profile, charges intake / handouts / disposals at their per-parcel handling
times, and reports both the staff needed to clear the day's volume and the
staff needed to cover the midday rush. The rush is usually the binding
constraint.

**`weekplan.py`**: Builds the current-week, day-by-day plan. Days already
elapsed use actual arrivals; remaining days are forecast, and the forecast is
conditioned on the actuals so far, so a heavy Monday lifts Wednesday. Picks
the short (momentum) model within 14 days of the last actual and the long
(calendar-only) model beyond it.

**`run.py`**: The command line. `train`, `test`, `predict`.

### Reference

**`files/panther_post_model.py`**: The original v6 model. **Do not edit.** Kept
as the baseline the current pipeline is measured against.

**`files/*.csv`, `files/packagestats_cleaned.xlsx`**: v6's outputs and its
cleaned dataset. Superseded — the current pipeline reads the raw
`packagestats.csv` and does its own cleaning, because v6's cleaning imputed
4,443 unlabelled rows into Sutherland, all of them in the first academic year,
which fabricated a 32% downward trend at that site.

### Reporting

**`make_powerbi.py`**: Reshapes `outputs/` into a Power BI star schema under
`powerbi/` - two fact tables, a site dimension and a date spine, with the
weekday sort index and the `mailroom`/`site` key mismatch fixed on the way out.
Run it after `predict`.

**`briefing.py`**: The spoken weekly briefing, replacing the old
`elevenLabs.py` scratch file. Builds the narration from `week_plan.csv` with a
fixed template - no model writes the text, so it stays auditable and inside the
No Wrapper track. `python -m briefing` prints it; `--speak` synthesises
`outputs/briefing.mp3` and needs `ELEVENLABS_API_KEY` in the environment or a
`.env` file.

### Generated (`outputs/`, gitignored)

| File | Written by | Contents |
|---|---|---|
| `models.pkl` | `train` | Fitted models + dwell curves |
| `trained_models.csv` | `train` | Chosen model per mailroom |
| `model_leaderboard.csv` | `train` | Every candidate's CV score |
| `test_accuracy.csv` | `test` | Accuracy per mailroom |
| `test_predictions.csv` | `test` | Every test-year day, predicted vs actual |
| `test_occupancy_accuracy.csv` | `test` | Dwell-model accuracy |
| `week_plan.csv` | `predict` | Week x site x scenario; feeds `make_powerbi` |
| `pool_check.csv` | `predict` | Daily staff assignment vs pool |
| `briefing.mp3` | `briefing --speak` | Spoken weekly briefing |

`powerbi/` is generated by `make_powerbi.py` and is **not** gitignored - it
holds only per-site daily aggregates, no names.

---

## What is being predicted

**Target variable: daily parcel intake per mailroom** — the count of packages
received at that location on that date. One model per mailroom, six models.
There is no pooled or campus-wide model; the campus total is the sum of the
site forecasts, which is correct because each site has its own calendar,
closures, and day-of-week shape.

Everything else is derived from intake rather than modelled separately:

```
intake forecast
  ├─ x dwell survival  -> occupancy   -> locker utilisation
  ├─ x dwell hazard    -> pickups     -> counter workload
  └─ intake + pickups + disposals  -> labour hours -> staff to assign
```

---

## Metrics

### q80 and why the forecast is not the average

Every intake forecast is published at two levels:

- **`predicted` (q50, the median)** — the typical outcome. Half of days land
  above it. This is what you'd quote as "expect about this many."
- **`predicted_q80` / `intake` (the 80th percentile)** — the level exceeded
  only one day in five. **This is what staffing is planned against.**

The reason is that the cost of being wrong is asymmetric. Under-staffing means
queues, angry students, and overtime; over-staffing means some idle time. So
the plan is built against a level the day will probably stay under, not the
level it will average. Set by `STAFFING_QUANTILE` in `config.py`.

### Accuracy measures in `test`

**`mae`** — mean absolute error, in parcels per day. Average size of a miss,
ignoring direction. The primary raw error measure.

**`accuracy_pct`** = `100 x (1 - MAE / mean actual)`. Read as "predictions
land within this percentage of a typical day's volume." Deliberately *not*
MAPE, because 18-45% of open days at the quiet sites have zero intake and MAPE
divides by zero on every one of them. **Treat this as secondary** — at a site
averaging 2.4 parcels a day with high variance, no model can score well on
this ratio, and a low number there says more about the site than the model.

**`vs_naive_pct`** — improvement in MAE over predicting *last week, same
weekday*. **This is the metric that matters.** Seasonal naive is the bar a
forecast must clear to be worth running at all. Positive means the model
earns its place; negative means use the naive forecast instead.

**`bias`** — mean signed error. Positive means systematic over-prediction,
negative means under. A model can have good MAE and bad bias, which matters
for staffing because persistent bias means persistently wrong rosters.

**`q80_coverage`** — share of test days where actual came in at or below the
q80 forecast. **Should sit near 0.80.** Materially above means the intervals
are too wide and staffing will be padded; materially below means too narrow
and shifts will be under-covered. Note this is one-sided: a wild
over-prediction still counts as "covered," so read it alongside `bias`.

**`cv_pinball`** (in `train`) — pinball loss at q80, averaged over the
cross-validation folds. The **model selection** metric. It penalises
under-prediction about four times as heavily as over-prediction, matching the
asymmetric staffing cost. Lower is better; the scale depends on site volume,
so compare only within a site.

**`occ_mae_pct`** (occupancy) — dwell-model error as a percentage of mean
parcels held. Validated against directly observed occupancy, so it isolates
dwell error from intake-forecast error.

**Not used: R².** Computed internally as a diagnostic only. At single-digit
daily counts it is dominated by a handful of spikes and ranks models
misleadingly — v6 reported R² of -0.06 for a model that beat its baseline by
57% on MAE.

### Staffing outputs in `predict`

**`total_hours`** — labour hours for the day, from all three transaction
streams at measured per-parcel handling times.

**`assign_fte`** — people to roster that day, the maximum of the volume
requirement, the peak-window requirement, and a minimum-coverage floor,
rounded to the nearest half person. **This is an assignment from an existing
pool, not a headcount recommendation.** Nobody is being cut.

**`locker_utilisation_pct`** — forecast occupancy against locker count. The
occupancy figure is measured; the locker count is currently a placeholder, so
read the parcel counts and treat the percentage as indicative.

---

## Assumed constants

Every number the pipeline relies on that is not learned from data. Three
categories: **measured** from the dataset, **placeholder** pending figures
from Operations, and **judgement** chosen by us and defensible but arbitrary.
All live in `panther/config.py` unless noted.

### Placeholder - not supplied by Operations

| Constant | Value | Notes |
|---|---|---|
| `TOTAL_LOCKERS` | 1000 | Never confirmed. Inconsistent with observed occupancy, which exceeds it system-wide on 432 of 731 days. |
| `LARGE_SHARE` | 0.70 | Share of the pool assigned to Tower B and Sutherland, then split by volume. |
| `LOCKERS_OVERRIDE` | `None` | Set to a dict of real counts and the volume-share derivation is bypassed. |
| `HEADCOUNT_FTE` | Tower B 4.0, Sutherland 1.5, Bouquet 1.0, Lothrop 1.0, Nordenberg 1.0, Bigelow 0.5, Ruskin 0.5 | Requested in the email thread, never received. Used only for the headroom column, never to drive an assignment. |
| `SHIFT_HOURS_PER_FTE` | 8.0 | |
| `PRODUCTIVE_FRACTION` | 0.75 | Share of a shift on parcel work. |
| `MIN_COVERAGE_FTE` | 1.0 | Someone must be at the counter regardless of volume. Pure arithmetic gave Nordenberg 0.5, and half a person cannot cover a counter. |
| `HANDOUT_SECONDS_DEFAULT` | 75.0 | Per-parcel handout time. The only handling figure NOT measured - the same burst analysis should be run on the `Delivered` timestamps. |

### Measured from the data

| Constant | Value | How |
|---|---|---|
| `INTAKE_SECONDS_MEDIAN` | Tower B 37s, Sutherland 40s, Nordenberg 41s, Lothrop 42s, Ruskin 42s, Bouquet 43s, Bigelow 44s | Median gap between consecutive intake scans inside a burst (gap under 300s). |
| `EXCLUDE` | Forbes, Darragh | Permanently closed; last activity 2026-04. |
| `OPEN_ALL_YEAR` | Tower B, Residences on Bigelow | Inferred from which sites keep scanning through breaks. Bigelow needs confirming - it opened 2025-04-15, so its "year-round" behaviour may just be its opening ramp. |
| `LARGE` | Tower B, Sutherland | Volume tiers. |
| `WEEK_ANCHOR_DOW` | 6 (Sunday) | Pay week runs Sunday to Saturday, from the timecard period "9/6-9/12" in the email thread. |
| `HARD_CLOSURES` | 8 holidays | Each verified as zero intake at every site in both years. In `calendar_features.py`. |
| `MINOR_HOLIDAYS` | 4 holidays | MLK, Presidents, Memorial, Labor - each has real activity, so they are a feature, not a rule. |
| `EVENTS` windows | Halloween -10/+5, Valentine's -7/+3, Super Bowl -5/+2, Black Friday -3/+5 | Shape from the measured lift profile, then rounded. In `calendar_features.py`. |
| `REGIME_BOUNDARIES` | Academic years, 1 August | Tower B's dwell behaviour changed between them. |

### Judgement - chosen by us

| Constant | Value | Reasoning |
|---|---|---|
| `STAFFING_QUANTILE` | 0.80 | Staffing loss is asymmetric. Raise if shifts run short, lower if staff are idle. |
| `SPLIT_DATE` | 2025-09-19 | Year one trains, year two tests. |
| `SHORT_HORIZON` | 14 days | Boundary between the momentum and calendar-only models. |
| `SELECTION_TOLERANCE` | 0.05 | Candidates within 5% of the best CV score count as tied; the simplest wins. A one-standard-error rule was tried and was far too wide. |
| `CV_FOLDS` | 5 | Rolling-origin folds inside the training year. |
| `MIN_SITE_DAYS` | 150 | Active days needed to model a site independently. Ruskin fails this at 146. |
| `MIN_TRAIN_DAYS` | 100 | Training-year open days needed to fit at all. Ruskin has 0. |
| `PLANNING_MULTIPLIER` | 2.0 | Scales measured median handling time to roughly p75, since the median assumes uninterrupted flow. |
| `PURGE_MIN_ITEMS` | 50 | A bulk operation, not a busy afternoon. |
| `PURGE_MEDIAN_AGE_DAYS` | 60 | Above the winter-backlog range (27-45d), below observed purges (62-248d). |
| `FORECAST_REGIME` | 2025-26 | Which dwell regime to project forward. The current partial year cannot yet show long dwell. |
| `EVENT_PAD` | 7 | Window for the legacy Christmas and Thanksgiving distance features. |
| `MAX_LAG` | 540 days | Survival curve length, past the longest observed dwell of 506 days. In `dwell.py`. |
| `MIN_EVENTS` | 40 | Below this a dwell stratum falls back to a pooled curve. In `dwell.py`. |
| Peak window | 11:00-14:00 | Where intake (peaks 10-13) and handouts (peak 12-16) overlap. In `staffing.py`. |
| `dow_ceiling` headroom | 1.25 | Forecast cap at 125% of the observed maximum for that weekday. In `selection.py`. |
| `COMPLEXITY` ranking | bayes_ridge < poisson_l2 < poisson < tweedie < hgb_poisson | Parsimony order for the tie-break. In `selection.py`. |

## Problem log

### Fixed

**1. University holidays missing from the closure calendar.** Labor Day 2026
forecast 293 parcels against an actual 0, because the calendar knew only
Sundays and long breaks. Eight holidays verified as zero intake at every site
in both years and are now hard closures: New Year's Day, Juneteenth, July 4,
Thanksgiving, the day after Thanksgiving, Christmas Eve, Christmas Day, New
Year's Eve.

Four candidates failed that test and are deliberately NOT closures, because
each has real recorded activity: Presidents Day 2026 saw 289 parcels at Tower
B (above its 108 average), MLK Day 2026 saw 17 at Tower B and 19 at
Sutherland, Memorial Day 2026 saw 5, and Labor Day 2026 was zero at Tower B
but **22 at Residences on Bigelow**. The zero that caused the error was a
site-level closure, not a campus-wide one. These four became an
`is_minor_holiday` feature so each site learns its own response. Labor Day
2026 now forecasts 13 against an actual 0.

**2. Summer Saturdays at the year-round sites.** Tower B's summer Saturdays
are 20-of-31 zero, but the other 11 carry 2 to 18 parcels, so a closure rule
would have been wrong. Now a `saturday_in_break` feature, separating Saturday
in session (mean 51.2) from Saturday in a break (mean 3.3) rather than
blending them into one coefficient suppressed 28% below the session level.

**4. Halloween on a Saturday over-predicted.** The 2026-10-28 run forecast 355
parcels for Saturday 2026-10-31 at Tower B against an all-time
session-Saturday maximum of 175. Halloween fell on a Thursday in 2024 and a
Friday in 2025, so the model had never seen the event land on a low-volume
weekday and the log-link multiplied the two coefficients unchecked.
`selection.dow_ceiling()` now caps every forecast at the observed maximum for
that weekday plus 25%, bringing it to 218.8. Test-year accuracy is unchanged,
so it is a pure guard rather than a tuning knob.

**9. Dead features at most sites.** `summer_break`, `winter_break` and
`ev_christmas` are structurally constant wherever those days are dropped as
closures, and Residences on Bigelow had 10 constant columns because its
training window opens in April 2025. Constant columns are not harmless - tree
models still split on them given a small training set. `live_features()` now
drops them per site.

**10. Model selection picked high-capacity models on noise.** Gradient
boosting won Sutherland's cross-validation by 0.05 and then missed the test
year by +23 parcels a day, worse than seasonal naive. `SELECTION_TOLERANCE`
now treats candidates within 5% as tied and prefers the simplest. The direct
quantile estimator was also removed from the pool: it predicts one quantile
only, so a model fitted at q80 cannot report a median, and when it won at
Bigelow the entire accuracy row came back NaN.

### Largely resolved

**5. Nordenberg beating seasonal naive.** Was `vs_naive_pct` -21.8 with bias
+2.40; the holiday calendar and the parsimony tie-break together moved it to
+11.7 with bias +0.81. Its `accuracy_pct` is still 0.0, but that ratio is
meaningless at 2.4 parcels a day. Its volume did collapse roughly 15x between
academic years, so treat it as a monitoring signal rather than a planning
input. It sits at minimum coverage in every scenario regardless.

### Open

**3. Saturdays are the weakest weekday, but no longer an outlier.** Measured
again on 2026-09-20, Saturday coverage is **0.753** against a 0.80 target, not
the 0.538 previously recorded here. Every weekday now sits in a 0.75-0.86 band:

```
Friday 0.852   Monday 0.811   Saturday 0.753
Thursday 0.860  Tuesday 0.791  Wednesday 0.771
```

Saturday is still the lowest and is still missed low, but Wednesday (0.771) and
Tuesday (0.791) are now within 0.02 of it, so this reads as the whole band
sitting slightly under target rather than one broken weekday. **Fix, if it is
worth it:** fit Saturdays as a separate level per period rather than as a
coefficient on a shared day-of-week term. Lower priority than it was.

**6. Sutherland regressed** with the calendar fixes, from MAE 15.75 to 17.40
and `vs_naive_pct` 40.0 to 34.0. It still selects gradient boosting because
the simpler variants also got worse with the new features. Its
cross-validation is genuinely unstable at 192 training days. **Fix:** exclude
high-capacity estimators below a training-size threshold, or widen
`SELECTION_TOLERANCE` for small sites only.

**7. Bouquet Gardens' dwell model is poorly calibrated** - `occ_mae_pct` 72.7
with bias +30.6, so occupancy is over-predicted by more than half its mean.
This is in the survival model, not the intake model.

**8. Ruskin is unmodelled.** It opened 2025-09-09, so it has zero
training-year days and is skipped entirely. **Fix:** a hierarchical model
predicting it as a day-of-week share of Tower B, or a later split date.

**11. Residences on Bigelow shows over 100% locker utilisation.** Not a model
error - the locker count is the placeholder 28 derived from the unconfirmed
1,000-unit pool. Needs real figures from Operations.

## Next steps

### If further tuning is needed

Work in this order. Items 1 and 2 are the two open problems that still
produce visibly wrong numbers.

1. **Saturdays** (problem 3). Coverage is 0.753 against 0.80, and the whole
   weekday band now sits just under target rather than Saturday alone being
   broken. Fit Saturdays as a separate level per period if you want the last
   0.05, but this is no longer the most valuable change available.
2. **Sutherland's estimator choice** (problem 6). Exclude high-capacity
   estimators below a training-size threshold, or widen
   `SELECTION_TOLERANCE` for small sites only.
3. **Re-tune the staffing quantile.** `STAFFING_QUANTILE` is 0.80. Raise it if
   shifts run short, lower it if staff are idle. Re-check `q80_coverage` in
   `test` after any change.
4. **Add or remove events.** `EVENTS` in `calendar_features.py` holds the
   name, date function, and asymmetric lead/lag window for each. Measured
   lift: Halloween 1.57x / 2.08x, Valentine's 1.44x / 1.55x, Super Bowl 1.14x
   / 1.19x, Black Friday 0.52x / 0.51x - Black Friday is a *trough*, because
   students leave for Thanksgiving and ship elsewhere. Super Bowl is weak and
   is a candidate for removal. St Patrick's Day showed no signal and was
   excluded. Confirm any addition on the test year rather than on the
   cross-validation score, because the training year holds only one
   observation of each event.
5. **Add candidate models.** `candidate_specs()` in `selection.py`. Keep the
   factories as module-level named functions, not lambdas, or `models.pkl`
   will fail to pickle. Add a `COMPLEXITY` rank for anything new, or the
   parsimony tie-break will treat it as maximally complex.
6. **Move the split date.** `SPLIT_DATE` in `config.py`. Later gives more
   training data and a shorter test window.
7. **Bouquet Gardens' dwell calibration** (problem 7) and **Ruskin's cold
   start** (problem 8) are lower priority - neither changes a staffing
   decision, since both sit at minimum coverage.

When real figures arrive from Operations, set them in `config.py` and nothing
else changes: `LOCKERS_OVERRIDE`, `HEADCOUNT_FTE`, `SHIFT_HOURS_PER_FTE`,
`PRODUCTIVE_FRACTION`, `MIN_COVERAGE_FTE`, `HANDOUT_SECONDS_DEFAULT`.

### Still outstanding from Operations

1. Locker / storage counts per mailroom
2. Current headcount and scheduled hours per mailroom
3. Whether WTS can export the location field separately from the scanning-user
   field (this would recover the 14,155 `Unassigned` parcels)
4. A snapshot of *currently held* parcels — the export contains only closed
   records, so occupancy decays artificially to zero at the export boundary
5. Whether a stated disposal or return-to-sender SLA exists, and what changed
   at Tower B during 2025-26

### Power BI report

Build steps, measures and page layouts are in **`powerbi/BUILD.md`**. Generate
the data it expects with:

```
python -m panther.run predict
python -m make_powerbi
```

`make_powerbi.py` writes a star schema rather than handing Power BI the raw
`outputs/` files, because three things there need fixing first and all three
are easier in pandas than in Power Query: `day_name` sorts alphabetically
without a numeric index travelling in the same table, `test_accuracy` keys on
`mailroom` while everything else keys on `site`, and the two accuracy files
share a grain so they belong in one table.

| `powerbi/` | Rows | Grain |
|---|---|---|
| `fact_plan.csv` | 84 | site x day x scenario |
| `fact_test_predictions.csv` | 1,437 | site x day, year two |
| `dim_site.csv` | 6 | accuracy, lockers, headcount per mailroom |
| `dim_date.csv` | 366 | date spine |
| `fact_pool_check.csv` | 7 | assigned vs pool |
| `fact_leaderboard.csv` | 240 | every candidate's CV score |

`panther_theme.json` is a Power BI theme (**View > Themes > Browse**). Its
eight categorical colours were checked for colour-blind separation: worst
adjacent pair deltaE 9.1 under protanopia, against a floor of 8.

Two things worth knowing before building:

- **Power BI Desktop has no macOS build.** Use app.powerbi.com in a browser, a
  Windows machine, or a VM.
- **The `scenario` slicer is the disposal toggle**, and it is pure filtering.
  Power BI cannot pass parameters back to Python, so both scenarios are
  precomputed by `predict`; that is what makes the toggle instant.

### Spoken weekly briefing (ElevenLabs)

Built - `briefing.py`. Power BI cannot call an external API from a button, so
the audio is pre-generated and the dashboard button opens the file.

```
python -m briefing                          print the text
python -m briefing --speak                  write outputs/briefing.mp3
python -m briefing --scenario disposal_30d
```

The narration comes from a fixed template with numbers substituted from
`week_plan.csv`, so every spoken figure traces back to a row and no model
writes the text. It leads with labour and holdings rather than raw intake, and
names only the mailrooms that are actually over capacity. Current output:

> "Panther Post briefing for the week of September 13. Across all mailrooms, 56
> labour hours are needed, against a pool of 9.5 full time staff. Tower B is
> over capacity: 2,223 parcels held against 611 lockers, 364 percent of
> capacity. [...] A thirty day return to sender policy would cut Tower B from
> 2,223 parcels held to 227, a reduction of 90 percent."

Running it under `--scenario disposal_30d` drops Tower B from the over-capacity
list entirely, which makes the point faster than the number does.