"""
Labour-hour and headcount translation.

Staffing is driven by TRANSACTIONS, not inventory. Three streams generate
work, and each is charged separately:

    intake    - scanning parcels in; carrier-driven, front-loaded
    handouts  - releasing parcels to students; counter work, cannot be
                time-shifted or deferred to a quieter hour
    disposals - processing returns under a hold deadline; only non-zero in the
                disposal scenario, and deliberately charged for, because
                "enforce a 30-day policy" is not free labour

Per-parcel handling times are measured from the scan timestamps themselves
(median gap between consecutive scans inside a burst), so no time-motion study
is required. Planning uses roughly the p75 rather than the median, since the
median assumes uninterrupted flow with no walking, interruptions, or customer
contact.

Two numbers come out, and they answer different questions:

    required_fte  - staff needed to clear the day's volume within a shift.
                    Answers "is the establishment big enough?"
    peak_fte      - staff needed to hold the busiest window without a queue.
                    Answers "when should they actually be rostered?"

peak_fte is normally the binding constraint and is the one worth showing a
manager, because intake peaks 10:00-13:00 while handouts peak 12:00-16:00 and
the two overlap at midday.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as cfg

# Window used for the peak-load calculation, from the measured intra-day
# curves: intake concentrates 49% of daily volume in 10:00-13:00, handouts 54%
# in 12:00-16:00. 11:00-14:00 is where the two stack.
PEAK_START_HOUR = 11
PEAK_END_HOUR = 14


def intraday_profile(tx: pd.DataFrame) -> pd.DataFrame:
    """Share of each day's intake and handouts falling in each hour, per site.

    Measured rather than assumed. Used to size the peak window.
    """
    rows = []
    for site, g in tx[tx['site_is_known']].groupby('site', observed=True):
        intake_h = g['received'].dt.hour.value_counts(normalize=True)
        out_h = g['delivered'].dropna().dt.hour.value_counts(normalize=True)
        for hour in range(24):
            rows.append({
                'site': site,
                'hour': hour,
                'intake_share': float(intake_h.get(hour, 0.0)),
                'handout_share': float(out_h.get(hour, 0.0)),
            })
    return pd.DataFrame(rows)


def peak_shares(profile: pd.DataFrame) -> pd.DataFrame:
    """Fraction of daily intake / handout volume inside the peak window."""
    win = profile[(profile['hour'] >= PEAK_START_HOUR)
                  & (profile['hour'] < PEAK_END_HOUR)]
    return win.groupby('site')[['intake_share', 'handout_share']].sum() \
              .rename(columns={'intake_share': 'peak_intake_share',
                               'handout_share': 'peak_handout_share'})


def _seconds(site: str) -> tuple[float, float]:
    """Planning seconds per intake scan and per handout for a site."""
    intake = cfg.INTAKE_SECONDS_MEDIAN.get(site, cfg.INTAKE_SECONDS_DEFAULT)
    return intake * cfg.PLANNING_MULTIPLIER, cfg.HANDOUT_SECONDS_DEFAULT


def _fte_from_hours(hours: float) -> float:
    """Convert labour hours to FTE, rounded up to the nearest half person.

    Half-person granularity because a shift can realistically be split, but a
    quarter of a person cannot be rostered.
    """
    capacity = cfg.SHIFT_HOURS_PER_FTE * cfg.PRODUCTIVE_FRACTION
    if capacity <= 0:
        return float('nan')
    return math.ceil((hours / capacity) * 2) / 2


def daily_labour(projection: pd.DataFrame, peaks: pd.DataFrame) -> pd.DataFrame:
    """Per-site, per-day labour requirement from a dwell projection.

    `projection` is the output of DwellModel.project(): it must carry intake,
    pickups_hat, and disposals_hat. Forecast volumes should already be at the
    staffing quantile - staff to a high quantile, not the mean, because
    under-staffing costs queues and overtime while over-staffing costs mild
    idle time.
    """
    out = projection.copy()
    if 'disposals_hat' not in out.columns:
        out['disposals_hat'] = 0.0

    sec_intake, sec_handout = zip(*[_seconds(s) for s in out['site']])
    out['intake_hours'] = out['intake'] * np.asarray(sec_intake) / 3600.0
    out['handout_hours'] = out['pickups_hat'] * np.asarray(sec_handout) / 3600.0
    # A return is a handout plus paperwork; charge it at the handout rate.
    out['disposal_hours'] = out['disposals_hat'] * cfg.HANDOUT_SECONDS_DEFAULT / 3600.0
    out['total_hours'] = (out['intake_hours'] + out['handout_hours']
                          + out['disposal_hours'])

    out = out.merge(peaks, left_on='site', right_index=True, how='left')
    out[['peak_intake_share', 'peak_handout_share']] = \
        out[['peak_intake_share', 'peak_handout_share']].fillna(0.4)

    window_hours = PEAK_END_HOUR - PEAK_START_HOUR
    out['peak_window_hours'] = (
        out['intake_hours'] * out['peak_intake_share']
        + (out['handout_hours'] + out['disposal_hours']) * out['peak_handout_share']
    )
    # Bodies needed simultaneously during the window: labour-hours falling in
    # the window, spread over its length, grossed up for non-productive time.
    out['peak_fte'] = (
        out['peak_window_hours'] / window_hours / cfg.PRODUCTIVE_FRACTION
    ).apply(lambda x: math.ceil(x * 2) / 2)
    out['required_fte'] = out['total_hours'].apply(_fte_from_hours)

    keep = ['site', 'date', 'intake', 'pickups_hat', 'disposals_hat',
            'occupancy_hat', 'intake_hours', 'handout_hours', 'disposal_hours',
            'total_hours', 'peak_window_hours', 'required_fte', 'peak_fte']
    return out[keep]


def pay_week(dates: pd.Series) -> pd.Series:
    """Label each date with the Sunday that opens its pay week.

    Sunday-Saturday, inferred from the timecard exchange in the email thread
    (period '9/6-9/12'; 2026-09-06 is a Sunday). Aligning to this boundary
    means the output drops into the existing scheduling cadence without
    translation.
    """
    dt = pd.to_datetime(dates)
    return (dt - pd.to_timedelta((dt.dt.dayofweek + 1) % 7, unit='D')).dt.normalize()


def weekly_roster(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily labour into a Sunday-anchored staffing recommendation."""
    d = daily.copy()
    d['week_start'] = pay_week(d['date'])

    agg = d.groupby(['site', 'week_start']).agg(
        forecast_intake=('intake', 'sum'),
        forecast_handouts=('pickups_hat', 'sum'),
        forecast_disposals=('disposals_hat', 'sum'),
        peak_occupancy=('occupancy_hat', 'max'),
        total_hours=('total_hours', 'sum'),
        busiest_day_hours=('total_hours', 'max'),
        max_peak_fte=('peak_fte', 'max'),
        open_days=('date', 'count'),
    ).reset_index()

    # Three constraints, whichever binds hardest: clear the average day's
    # volume, cover the midday rush, and keep the counter staffed at all.
    agg['volume_fte'] = (agg['total_hours'] / agg['open_days'].clip(lower=1)) \
        .apply(_fte_from_hours)
    demand_fte = agg[['volume_fte', 'max_peak_fte']].max(axis=1)
    agg['recommended_fte'] = demand_fte.clip(lower=cfg.MIN_COVERAGE_FTE)
    agg['binding_constraint'] = np.select(
        [demand_fte < cfg.MIN_COVERAGE_FTE,
         agg['max_peak_fte'] > agg['volume_fte']],
        ['minimum_cover', 'peak_window'], default='daily_volume')

    # Comparison against establishment is shown ONLY as headroom on parcel
    # work. A negative number does NOT mean a site is overstaffed: the model
    # cannot see letter mail or desk cover (see cfg.LABOUR_SCOPE_NOTE), so
    # parcel-only demand is a floor, not a total.
    agg['current_fte'] = agg['site'].map(cfg.HEADCOUNT_FTE)
    agg['parcel_fte_headroom'] = (agg['current_fte'] - agg['recommended_fte']).round(1)
    agg['understaffed_for_parcels'] = agg['parcel_fte_headroom'] < 0
    agg['scope_note'] = cfg.LABOUR_SCOPE_NOTE

    for c in ('forecast_intake', 'forecast_handouts', 'forecast_disposals',
              'peak_occupancy', 'total_hours', 'busiest_day_hours'):
        agg[c] = agg[c].round(1)
    return agg.sort_values(['week_start', 'forecast_intake'], ascending=[True, False])