"""
Reshape outputs/ into a Power BI star schema in powerbi/.

    python -m make_powerbi        (after: panther.run train / test / predict)

Power BI can import outputs/ directly, but three things need fixing first and
all three are easier here than in Power Query:

  * day_name sorts alphabetically (Friday first) unless a numeric day index
    travels with it, and Sort-by-column needs that index in the same table.
  * test_accuracy keys on `mailroom` while everything else keys on `site`,
    so the relationships will not line up without a rename.
  * accuracy lives in two files that share a grain (one row per mailroom);
    merged here, they make one table visual instead of two.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from panther import config as cfg

OUT = Path(cfg.OUTPUT_DIR)
PBI = Path('powerbi')

DOW = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']


def _day_index(df: pd.DataFrame) -> pd.DataFrame:
    """Numeric weekday to sort day_name by, Sunday first (WEEK_ANCHOR_DOW)."""
    df['day_index'] = df['day_name'].map({d: i for i, d in enumerate(DOW)})
    return df


def main() -> None:
    if not (OUT / 'week_plan.csv').exists():
        raise SystemExit('No outputs/week_plan.csv. Run: python -m panther.run predict')
    PBI.mkdir(exist_ok=True)

    # ---- fact: the weekly operating plan, both scenarios -----------------
    plan = _day_index(pd.read_csv(OUT / 'week_plan.csv', parse_dates=['date', 'week_start', 'as_of']))
    # Split the q80 staffing level from the median so a chart can show both.
    plan['intake_q80'] = plan['intake']
    plan['headroom_fte'] = plan['current_headcount_fte'] - plan['assign_fte']
    plan.to_csv(PBI / 'fact_plan.csv', index=False)

    # ---- fact: test-year predicted vs actual -----------------------------
    pred = _day_index(pd.read_csv(OUT / 'test_predictions.csv', parse_dates=['date']))
    pred['month_name'] = pred['date'].dt.strftime('%b')
    pred['month_index'] = pred['date'].dt.month
    pred['q80_miss'] = (~pred['covered_by_q80']).astype(int)
    pred.to_csv(PBI / 'fact_test_predictions.csv', index=False)

    # ---- dim: one row per mailroom, accuracy + attributes ----------------
    acc = pd.read_csv(OUT / 'test_accuracy.csv').rename(columns={'mailroom': 'site'})
    occ = pd.read_csv(OUT / 'test_occupancy_accuracy.csv')
    trained = pd.read_csv(OUT / 'trained_models.csv').rename(columns={'mailroom': 'site'})

    site = acc.merge(occ, on='site', how='outer').merge(
        trained[['site', 'train_days', 'cv_pinball_q80', 'events_used']], on='site', how='outer')
    site['tier'] = site['site'].apply(lambda s: 'large' if s in cfg.LARGE else 'small')
    site['open_all_year'] = site['site'].isin(cfg.OPEN_ALL_YEAR)
    site['headcount_fte'] = site['site'].map(cfg.HEADCOUNT_FTE)
    site['lockers'] = site['site'].map(
        plan.drop_duplicates('site').set_index('site')['lockers'])
    # Positive vs_naive_pct is the bar a forecast must clear to be worth running.
    site['beats_naive'] = site['vs_naive_pct'] > 0
    site.to_csv(PBI / 'dim_site.csv', index=False)

    # ---- dim: date, spanning both fact tables ----------------------------
    lo = min(pred['date'].min(), plan['date'].min())
    hi = max(pred['date'].max(), plan['date'].max())
    dim = pd.DataFrame({'date': pd.date_range(lo, hi, freq='D')})
    dim['day_name'] = dim['date'].dt.day_name()
    dim = _day_index(dim)
    dim['month_name'] = dim['date'].dt.strftime('%b')
    dim['month_index'] = dim['date'].dt.month
    dim['year'] = dim['date'].dt.year
    dim['academic_year'] = dim['date'].apply(
        lambda d: f'{d.year}-{str(d.year + 1)[2:]}' if d.month >= 8
        else f'{d.year - 1}-{str(d.year)[2:]}')
    dim['is_weekend'] = dim['day_index'].isin([0, 6])
    dim.to_csv(PBI / 'dim_date.csv', index=False)

    # ---- fact: staff pool vs assignment ----------------------------------
    pool = _day_index(pd.read_csv(OUT / 'pool_check.csv', parse_dates=['date']))
    pool.to_csv(PBI / 'fact_pool_check.csv', index=False)

    # ---- fact: model leaderboard -----------------------------------------
    board = pd.read_csv(OUT / 'model_leaderboard.csv')
    board[['estimator', 'feature_set']] = board['model'].str.split('|', n=1, expand=True)
    board.to_csv(PBI / 'fact_leaderboard.csv', index=False)

    for f in sorted(PBI.glob('*.csv')):
        print(f'  {f}  ({len(pd.read_csv(f)):,} rows)')

    peak = plan.pivot_table(index='site', columns='scenario',
                            values='occupancy_hat', aggfunc='max')
    print('\n  peak parcels held this week, by scenario:')
    print(peak.round(0).to_string())


if __name__ == '__main__':
    main()
