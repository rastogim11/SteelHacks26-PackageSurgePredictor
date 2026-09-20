# Power BI dashboard — build steps

Every column named below was checked against the generated files. Nothing here
is guessed.

---

## 0. Windows quick start — you do NOT need Python

The six CSVs in this folder are **committed to the repo**. You do not need the
pipeline, a virtual environment, or the raw data file to build the report.
Clone and open:

```
git clone https://github.com/rastogim11/SteelHacks26-PackageSurgePredictor.git
```

Then open **Power BI Desktop** and go to step 1. That is the whole setup.

Install Power BI Desktop from the Microsoft Store (search "Power BI Desktop")
or powerbi.microsoft.com/desktop. The Store version auto-updates and needs no
admin rights, which is usually the faster route on a locked-down laptop.

You will *not* be able to run `python -m make_powerbi` without the raw
`packagestats.csv`, which is gitignored and stays on Manan's machine. You do
not need to. Only regenerate if the underlying forecast changes, and in that
case have Manan re-run it and push:

```
python -m panther.run predict && python -m make_powerbi
```

On macOS Power BI Desktop does not exist at all — use app.powerbi.com in a
browser instead. Most of the steps below apply; a few formatting panes differ.

---

## 1. Load

**Get Data > Text/CSV**, once per file in `powerbi/`:

| Table | Rows | Grain |
|---|---|---|
| `fact_plan` | 84 | site x day x scenario — the operating week |
| `fact_test_predictions` | 1,437 | site x day — year two, predicted vs actual |
| `dim_site` | 6 | one row per mailroom: accuracy, lockers, headcount |
| `dim_date` | 366 | date spine |
| `fact_pool_check` | 7 | staff assigned vs pool, per day |
| `fact_leaderboard` | 240 | every candidate model's CV score |

Check on import that `date`, `week_start` and `as_of` came in as **Date**, not
Text. If any arrived as Text, change the type in Power Query before loading.

## 2. Sort the weekday axis

Power BI sorts text alphabetically, so an unsorted `day_name` axis starts on
Friday. In **Model view**, for each of `fact_plan`, `fact_test_predictions`,
`dim_date` and `fact_pool_check`: select `day_name` > **Column tools** >
**Sort by column** > `day_index`. The index is already in all four tables,
Sunday = 0, matching the Sunday-to-Saturday pay week.

## 3. Relationships

In **Model view**, drag to create these. All are many-to-one, single direction.

```
dim_date[date]  ->  fact_plan[date]
dim_date[date]  ->  fact_test_predictions[date]
dim_date[date]  ->  fact_pool_check[date]
dim_site[site]  ->  fact_plan[site]
dim_site[site]  ->  fact_test_predictions[site]
dim_site[site]  ->  fact_leaderboard[site]
```

Then **Table tools > Mark as date table** on `dim_date`, using `date`.

## 4. Measures

New measure on `fact_plan`:

```DAX
Weekly Hours = SUM(fact_plan[total_hours])

Staff Days Assigned = SUM(fact_plan[assign_fte])

-- Occupancy is a stock, not a flow. Summing it across dates is meaningless,
-- so peak is a MAX over days and a SUM only across sites within one day.
Peak Held = MAX(fact_plan[occupancy_hat])

Peak Held Campus =
MAXX( VALUES(fact_plan[date]), CALCULATE(SUM(fact_plan[occupancy_hat])) )

-- Locker count is constant per site, so it must never be summed over days.
Lockers = SUMX( VALUES(fact_plan[site]), CALCULATE(MAX(fact_plan[lockers])) )

Peak Utilisation % = MAX(fact_plan[locker_utilisation_pct])

Peak Held Current  = CALCULATE([Peak Held], fact_plan[scenario] = "current")
Peak Held Disposal = CALCULATE([Peak Held], fact_plan[scenario] = "disposal_30d")

Disposal Reduction % =
DIVIDE( [Peak Held Current] - [Peak Held Disposal], [Peak Held Current] )
```

New measure on `fact_test_predictions`:

```DAX
Test MAE = AVERAGE(fact_test_predictions[abs_error])

Bias = AVERAGE(fact_test_predictions[error])

-- q80_miss is 1 when the day broke through the q80 band, so this is coverage.
-- Target 0.80. Recomputed from rows, so it responds to every slicer.
q80 Coverage = 1 - AVERAGE(fact_test_predictions[q80_miss])

Naive MAE =
AVERAGEX( fact_test_predictions,
          ABS(fact_test_predictions[actual] - fact_test_predictions[seasonal_naive]) )

-- The headline metric: beating "same weekday last week" is the bar a forecast
-- must clear to be worth running. Recomputed rather than averaging the
-- per-site percentages in dim_site, which would be wrong under a slicer.
vs Naive % = DIVIDE( [Naive MAE] - [Test MAE], [Naive MAE] )
```

Set format: `Disposal Reduction %`, `q80 Coverage` and `vs Naive %` to
Percentage, 1 decimal. The rest to whole or 1-decimal numbers.

## 5. Pages

### Page 1 — The Week

Slicers across the top: `fact_plan[scenario]`, `dim_site[site]`,
`fact_plan[source]`.

- **Cards**: `Weekly Hours`, `Staff Days Assigned`, `Peak Held Campus`,
  `Peak Utilisation %`
- **Clustered column** — Axis `day_name`, Values `intake`, Legend `source`.
  Actual days and forecast days get separate colors, so the "we are three days
  into the week" story is visible rather than asserted.
- **Matrix** — Rows `site`, Columns `day_name`, Values `assign_fte`.
  Conditional-format the background on a blue sequential ramp.
- **Line** — Axis `date`, Values `occupancy_hat`, Legend `site`.

### Page 2 — The Disposal Finding

This is the page that carries the argument, so give it room.

- **Line** — Axis `date`, Values `occupancy_hat`, Legend `scenario`, filtered to
  Tower B. Add an **Analytics > Constant line** at 611 labelled "lockers".
- **Card** — `Disposal Reduction %`. At Tower B it reads 89.8%.
- **Bar** — Axis `site`, Values `Peak Held Current` and `Peak Held Disposal`.
- **Text box**, stated plainly: Tower B's peak holdings fall from 2,223 to 227
  parcels under a 30-day return-to-sender policy. The constraint is
  non-collection, not demand, and the fix costs nothing.
- **Caveat text box**: locker counts are a placeholder pending Operations, so
  read the parcel counts and treat the percentages as indicative.

### Page 3 — Does The Model Work

- **Line** — Axis `date`, Values `actual` and `predicted`, with `predicted_q80`
  as a third line. Slicer on `site`. Default it to Tower B.
- **Table** from `dim_site`: site, model, mae, vs_naive_pct, bias,
  q80_coverage. Conditional-format `vs_naive_pct` green above zero.
- **Column** — Axis `day_name`, Values `q80 Coverage`, with a constant line at
  0.80. This is the honest page: Saturday sits lowest.
- **Table** from `dim_site`: site, occ_mae_pct, occ_bias — dwell-model accuracy,
  validated against directly observed occupancy.

### Page 4 — Model Selection (optional, judges like it)

- **Scatter** from `fact_leaderboard` — X `cv_mae`, Y `cv_pinball`, Legend
  `estimator`, Play/slicer on `site`, with `selected` driving marker size.
  Shows that the winner was chosen by cross-validation, not by hand.

## 6. Theme

**View > Themes > Browse for themes** > `powerbi/panther_theme.json`.

Eight-slot categorical palette, colorblind-checked: worst adjacent pair
CVD deltaE 9.1, normal-vision 19.6, both clear of the floor. Three of the eight
sit below 3:1 contrast on white, so keep data labels visible on any visual that
uses slots 3, 4 or 5 alone.

## 6b. Checkpoint — confirm the load worked

Before building visuals, check these against what Power BI shows. If any is
off, the import went wrong rather than the model.

| Check | Expected |
|---|---|
| `fact_plan` row count | 84 |
| `scenario` distinct values | `current`, `disposal_30d` |
| `Weekly Hours` (scenario = current) | 56 |
| `Peak Held`, Tower B, current | 2,223 |
| `Peak Held`, Tower B, disposal_30d | 227 |
| `dim_site` row count | 6 |
| `vs_naive_pct` | positive for all 6 sites |

The CSVs are pure ASCII with no special characters, so no encoding option is
needed on import. If dates arrive as Text, set them in Power Query before
loading — do not fix it downstream.

## 7. Honesty notes for the demo

Judges reward knowing your own limits. Worth having on a slide:

- **Ruskin is missing** from all six-site visuals. It opened 2025-09-09, so it
  has no training-year data and is skipped rather than guessed at.
- **Locker counts are placeholders** derived from an unconfirmed 1,000-unit
  pool. Bigelow shows over 100% utilisation for that reason alone.
- **Saturday q80 coverage is 0.753** against a 0.80 target — the weakest
  weekday, and the one open modelling problem.
- **accuracy_pct is not the headline.** At Nordenberg's 2.4 parcels a day the
  ratio is meaningless. `vs_naive_pct` is the metric that matters, and all six
  sites are positive: +11.7 to +41.2.
