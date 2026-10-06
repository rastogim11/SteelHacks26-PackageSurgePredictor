"""
Build a multi-week dashboard dataset.

    python -m build_dashboard_data             this week + next week (default)
    python -m build_dashboard_data --weeks 3   this week + two ahead
    python -m build_dashboard_data --weeks 1   this week only

`panther.run predict` plans ONE week at a time, because that is the operational
unit - a roster is published weekly. A dashboard wants more than that: the week
just gone is mostly `actual`, which is what shows the forecast being right,
while the week ahead is entirely `forecast`, which is what makes it useful.
Carrying both lets one chart show the model being checked and the model being
used, with the existing `source` column telling them apart.

Each week is a separate `predict` run, so nothing about the forecasting changes
- this only concatenates the plans and hands them to make_powerbi.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from panther import config as cfg

OUT = Path(cfg.OUTPUT_DIR)
PLAN = OUT / 'week_plan.csv'


def _latest_actual() -> pd.Timestamp:
    """The last day with real data - the anchor for 'this week'."""
    from panther import data as dat
    for c in (cfg.DATA_RAW, cfg.DATA_CLEANED):
        if c and Path(c).exists():
            return dat.load_transactions(c)['recv_date'].max()
    raise SystemExit('No data file found. Run from the repo root.')


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog='build_dashboard_data')
    ap.add_argument('--weeks', type=int, default=2,
                    help='how many weeks to include, starting with the current one')
    args = ap.parse_args(argv)
    if args.weeks < 1:
        raise SystemExit('--weeks must be at least 1')

    anchor = _latest_actual()
    print(f'  latest actual: {anchor.date()}')

    frames = []
    for i in range(args.weeks):
        as_of = anchor + pd.Timedelta(days=7 * i)
        print(f'\n  [{i + 1}/{args.weeks}] predict --as-of {as_of.date()}')
        r = subprocess.run(
            [sys.executable, '-m', 'panther.run', 'predict', '--as-of', str(as_of.date())],
            capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-2000:])
            raise SystemExit(f'  predict failed for {as_of.date()}')

        wk = pd.read_csv(PLAN, parse_dates=['date', 'week_start', 'as_of'])
        # Label weeks relative to now, so the dashboard can say "This week" and
        # "Next week" rather than making the reader decode a date.
        wk['week_label'] = ('This week' if i == 0
                            else 'Next week' if i == 1
                            else f'Week +{i}')
        wk['week_offset'] = i
        frames.append(wk)
        span = f'{wk["date"].min().date()} to {wk["date"].max().date()}'
        openrows = wk[(wk['scenario'] == 'current') & wk['is_open']]
        print(f'      {span}   {len(wk)} rows   '
              f'{openrows["intake"].sum():.0f} parcels forecast')

    combined = pd.concat(frames, ignore_index=True)

    # A later week re-forecasts days an earlier run already covered only when
    # the windows overlap, which they do not here - but guard anyway, keeping
    # the earliest run for any duplicate, since it has more actuals.
    before = len(combined)
    combined = combined.drop_duplicates(
        subset=['scenario', 'site', 'date'], keep='first')
    if len(combined) != before:
        print(f'\n  dropped {before - len(combined)} duplicate site-days')

    combined.to_csv(PLAN, index=False)
    print(f'\n  combined -> {PLAN}   ({len(combined)} rows, '
          f'{combined["date"].nunique()} distinct days)')

    import make_powerbi
    print()
    make_powerbi.main()


if __name__ == '__main__':
    main()
