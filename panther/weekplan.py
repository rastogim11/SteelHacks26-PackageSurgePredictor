"""
Current-week day-by-day plan.

This is the dashboard's primary output: for the Sunday-Saturday week
containing `as_of`, one row per site per day carrying expected intake and the
number of people to put on that day.

Days already elapsed use ACTUAL intake. Remaining days are forecast, and the
forecast is conditioned on the actuals so far - the rolling-average features
are rebuilt from real arrivals up to `as_of`, so a heavier-than-expected
Monday raises Wednesday's estimate. That is the "arrival data in the week so
far" behaviour, and it is why the plan is worth regenerating nightly rather
than once on Sunday.

Staffing here is an ASSIGNMENT, not an establishment decision: given a pool of
employees, how many to roster on each day. Nobody is being cut. When the pool
cannot cover the day, the shortfall is surfaced rather than silently absorbed.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import calendar_features as cal
from . import config as cfg
from . import data as dat
from . import selection as sel
from . import staffing as stf


def week_bounds(as_of: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Sunday-Saturday week containing `as_of`."""
    as_of = pd.Timestamp(as_of).normalize()
    start = as_of - pd.Timedelta(days=(as_of.dayofweek + 1) % 7)
    return start, start + pd.Timedelta(days=6)


def _forecast_site_week(models: dict, site: str, week: pd.DatetimeIndex,
                        as_of: pd.Timestamp) -> pd.DataFrame:
    """Intake per day for one site: actuals up to as_of, forecast after.

    HORIZON-AWARE. Within SHORT_HORIZON days of the last actual, the short
    model is used and forecasting walks forward one day at a time, writing each
    prediction back so the rolling-average features stay populated - that
    feedback is what lets a heavy Monday lift Wednesday.

    Past SHORT_HORIZON the short model is abandoned entirely, because feeding
    it its own output compounds: over a 36-day fill Tower B's roll7 went from
    127 to 188,400 and the Poisson link overflowed to infinity. The long model
    is calendar-only, so it needs no history and cannot feed back on itself.
    """
    fitted = models['short']
    fitted_long = models['long']
    hist = fitted['frame'][['date', 'intake']].copy()
    hist = hist[hist['date'] <= as_of]
    last_actual = hist['date'].max() if len(hist) else None

    # A day is "actual" only if data exists for it. Keying off as_of alone
    # breaks whenever the requested week runs past the end of the data, which
    # is the normal case for a forward forecast.
    def is_known(d):
        return last_actual is not None and d <= last_actual

    open_week = [d for d in week
                 if not cal.is_closed(site, pd.Series([d])).iloc[0]]
    future = [d for d in open_week if not is_known(d)]

    # Open days between the last actual and the week itself must also be
    # filled, otherwise the rolling features have a hole in them.
    if future and last_actual is not None:
        bridge = pd.date_range(last_actual + pd.Timedelta(days=1),
                               min(future) - pd.Timedelta(days=1), freq='D')
        bridge = [d for d in bridge
                  if not cal.is_closed(site, pd.Series([d])).iloc[0]]
        future = bridge + future

    series = pd.concat([
        hist,
        pd.DataFrame({'date': future, 'intake': np.nan}),
    ], ignore_index=True).sort_values('date').reset_index(drop=True)

    def horizon_of(d):
        return (pd.Timestamp(d) - last_actual).days if last_actual is not None else 0

    def model_for(d):
        return fitted if horizon_of(d) <= cfg.SHORT_HORIZON else fitted_long

    first_future = len(series) - len(future)
    for i in range(first_future, len(series)):
        d = series['date'].iloc[i]
        if horizon_of(d) <= cfg.SHORT_HORIZON:
            feat = cal.add_features(series.iloc[:i + 1].copy(),
                                    fitted['t0'], fitted['tspan'],
                                    target='intake')
            row = feat.iloc[[-1]]
            val = float(sel.predict_quantile(
                fitted, row, cfg.STAFFING_QUANTILE)[0])
        else:
            # Calendar-only: build features from the date alone so nothing
            # downstream depends on the filled values above it.
            feat = cal.add_features(
                pd.DataFrame({'date': [d], 'intake': [np.nan]}),
                fitted_long['t0'], fitted_long['tspan'], target='intake')
            val = float(sel.predict_quantile(
                fitted_long, feat, cfg.STAFFING_QUANTILE)[0])
        series.loc[i, 'intake'] = 0.0 if not np.isfinite(val) else val

    rows = []
    for d in week:
        closed = bool(cal.is_closed(site, pd.Series([d])).iloc[0])
        if closed:
            rows.append({'date': d, 'is_open': False,
                         'source': 'closed',
                         'intake_q50': 0.0, 'intake_q80': 0.0,
                         'closed_reason': cal.closure_reason(site, d)})
            continue
        if is_known(d):
            actual = hist.loc[hist['date'] == d, 'intake']
            val = float(actual.iloc[0]) if len(actual) else 0.0
            rows.append({'date': d, 'is_open': True, 'source': 'actual',
                         'intake_q50': val, 'intake_q80': val,
                         'closed_reason': None})
        else:
            idx = series.index[series['date'] == d]
            m = model_for(d)
            if horizon_of(d) <= cfg.SHORT_HORIZON:
                feat = cal.add_features(series.loc[:idx[0]].copy(),
                                        m['t0'], m['tspan'], target='intake')
                row = feat.iloc[[-1]]
            else:
                row = cal.add_features(
                    pd.DataFrame({'date': [d], 'intake': [np.nan]}),
                    m['t0'], m['tspan'], target='intake')
            q50 = float(sel.predict_quantile(m, row, 0.50)[0])
            rows.append({
                'date': d, 'is_open': True, 'source': 'forecast',
                'horizon_days': horizon_of(d),
                'model_used': 'short' if horizon_of(d) <= cfg.SHORT_HORIZON else 'long',
                'intake_q50': 0.0 if not np.isfinite(q50) else round(q50, 1),
                'intake_q80': float(series.loc[idx[0], 'intake']),
                'closed_reason': None,
            })
    out = pd.DataFrame(rows)
    out.insert(0, 'site', site)
    return out


def build_week_plan(tx: pd.DataFrame, fitted_models: dict, dwell_model,
                    as_of: pd.Timestamp | None = None,
                    scenario: str = 'current') -> pd.DataFrame:
    """Day-by-day plan for the current week across all sites."""
    as_of = pd.Timestamp(as_of or tx['recv_date'].max()).normalize()
    start, end = week_bounds(as_of)
    week = pd.date_range(start, end, freq='D')

    peaks = stf.peak_shares(stf.intraday_profile(tx))
    frames = []

    for site, models in fitted_models.items():
        if 'short' not in models:          # tolerate an older cache
            models = {'short': models, 'long': models}
        wk = _forecast_site_week(models, site, week, as_of)

        # Occupancy needs the full arrival history, not just this week, since
        # parcels held from previous weeks are most of the standing stock.
        hist = dat.daily_panel(tx, site)
        hist = hist[hist['date'] < start][['date', 'intake']]
        combined = pd.concat([
            hist,
            wk.loc[wk['is_open'], ['date', 'intake_q80']]
              .rename(columns={'intake_q80': 'intake'}),
        ], ignore_index=True).sort_values('date')
        # Closed days still carry parcels; include them at zero intake.
        full = (combined.set_index('date')
                .reindex(pd.date_range(combined['date'].min(), end, freq='D'),
                         fill_value=0.0)
                .rename_axis('date').reset_index())

        proj = dwell_model.project(site, full['date'], full['intake'].to_numpy())
        wk = wk.merge(
            proj[['date', 'occupancy_hat', 'pickups_hat', 'disposals_hat']],
            on='date', how='left')
        frames.append(wk)

    plan = pd.concat(frames, ignore_index=True)
    plan[['occupancy_hat', 'pickups_hat', 'disposals_hat']] = \
        plan[['occupancy_hat', 'pickups_hat', 'disposals_hat']].fillna(0.0)
    plan = plan.rename(columns={'intake_q80': 'intake'})

    labour = stf.daily_labour(plan, peaks)
    plan = plan.merge(
        labour[['site', 'date', 'intake_hours', 'handout_hours',
                'disposal_hours', 'total_hours', 'peak_window_hours', 'peak_fte']],
        on=['site', 'date'], how='left')

    # Assignment: cover the day's volume AND the midday rush, but a closed day
    # needs nobody.
    volume_fte = plan['total_hours'].apply(
        lambda h: math.ceil((h / (cfg.SHIFT_HOURS_PER_FTE
                                  * cfg.PRODUCTIVE_FRACTION)) * 2) / 2)
    demand = np.maximum(volume_fte, plan['peak_fte'].fillna(0))
    plan['assign_fte'] = np.where(
        plan['is_open'], np.clip(demand, cfg.MIN_COVERAGE_FTE, None), 0.0)

    plan['scenario'] = scenario
    plan['week_start'] = start
    plan['as_of'] = as_of
    plan['day_name'] = plan['date'].dt.day_name()
    plan['intake'] = plan['intake'].round(1)
    plan['intake_q50'] = plan['intake_q50'].round(1)
    for c in ('occupancy_hat', 'pickups_hat', 'disposals_hat', 'total_hours',
              'peak_window_hours'):
        plan[c] = plan[c].round(1)

    for c in ('horizon_days', 'model_used'):
        if c not in plan.columns:
            plan[c] = None
    cols = ['scenario', 'week_start', 'as_of', 'site', 'date', 'day_name',
            'is_open', 'closed_reason', 'source', 'horizon_days', 'model_used',
            'intake_q50', 'intake',
            'occupancy_hat', 'pickups_hat', 'disposals_hat', 'total_hours',
            'peak_window_hours', 'peak_fte', 'assign_fte']
    return plan[cols].sort_values(['site', 'date']).reset_index(drop=True)


def pool_check(plan: pd.DataFrame) -> pd.DataFrame:
    """Daily total assignment against the available pool.

    Surfaces the days where the roster cannot be met from the pool rather than
    quietly scaling everyone down.
    """
    pool = sum(cfg.HEADCOUNT_FTE.values())
    by_day = plan.groupby('date').agg(
        assigned_fte=('assign_fte', 'sum'),
        sites_open=('is_open', 'sum'),
    ).reset_index()
    by_day['pool_fte'] = pool
    by_day['shortfall_fte'] = (by_day['assigned_fte'] - pool).clip(lower=0)
    by_day['day_name'] = by_day['date'].dt.day_name()
    return by_day