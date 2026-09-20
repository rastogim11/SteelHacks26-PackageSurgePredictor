"""
Campus calendar: closures, periods, and model features.

Closure and break logic is carried over unchanged from panther_post_model.py
v6 so that v7 numbers stay directly comparable against the baseline. The one
correction is SATURDAY, which v6 treated as a static dummy. Saturday's share
of intake falls 8.40% -> 4.83% -> 3.33% across the three academic years in the
data, so it is a trend, not a level: `sat_trend` carries the interaction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

# Period labels used to stratify dwell survival curves.
PERIODS = ('session', 'move_in', 'move_out', 'spring_break',
           'summer_break', 'winter_break')


def nth_weekday(year: int, month: int, weekday: int, n: int) -> pd.Timestamp:
    first = pd.Timestamp(year, month, 1)
    return first + pd.Timedelta(days=(weekday - first.dayofweek) % 7 + 7 * (n - 1))


def summer_mask(dt: pd.Series) -> pd.Series:
    m, d = dt.dt.month, dt.dt.day
    return ((m == 5) & (d >= 3)) | m.isin([6, 7]) | ((m == 8) & (d <= 16))


def winter_mask(dt: pd.Series) -> pd.Series:
    m, d = dt.dt.month, dt.dt.day
    return ((m == 12) & (d >= 18)) | ((m == 1) & (d <= 2))


def spring_mask(dt: pd.Series) -> pd.Series:
    m, d = dt.dt.month, dt.dt.day
    return (m == 3) & (d >= 5) & (d <= 15)


def move_in_mask(dt: pd.Series) -> pd.Series:
    m, d = dt.dt.month, dt.dt.day
    return ((m == 8) & (d >= 17)) | ((m == 9) & (d <= 10))


def move_out_mask(dt: pd.Series) -> pd.Series:
    m, d = dt.dt.month, dt.dt.day
    return ((m == 4) & (d >= 22)) | ((m == 5) & (d <= 2))


def sunday_mask(dt: pd.Series) -> pd.Series:
    return dt.dt.dayofweek == 6


def last_monday(year: int, month: int) -> pd.Timestamp:
    end = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    return end - pd.Timedelta(days=(end.dayofweek - 0) % 7)


# Holidays on which EVERY site recorded zero intake in BOTH years of data.
# These are structural closures: dropped from training, forced to zero at
# prediction. Each was verified against the data rather than assumed from a
# federal holiday list - see MINOR_HOLIDAYS for the ones that failed that test.
HARD_CLOSURES = {
    'new_years_day': lambda y: pd.Timestamp(y, 1, 1),
    'juneteenth':    lambda y: pd.Timestamp(y, 6, 19),
    'july_4':        lambda y: pd.Timestamp(y, 7, 4),
    'thanksgiving':  lambda y: nth_weekday(y, 11, 3, 4),
    'day_after_thanksgiving': lambda y: nth_weekday(y, 11, 3, 4) + pd.Timedelta(days=1),
    'christmas_eve': lambda y: pd.Timestamp(y, 12, 24),
    'christmas_day': lambda y: pd.Timestamp(y, 12, 25),
    'new_years_eve': lambda y: pd.Timestamp(y, 12, 31),
}

# Holidays that are NOT closures. Each has real recorded activity, so forcing
# a zero would trade one error for another:
#   Presidents Day 2026  289 parcels at Tower B - above its 108 average
#   MLK Day 2026          17 at Tower B, 19 at Sutherland
#   Labor Day 2026         0 at Tower B but 22 at Residences on Bigelow
#   Memorial Day 2026      5 at Tower B
# Labor Day is the instructive one: the zero at Tower B that produced a
# 293-parcel forecast error was real, but it was a site-level closure, not a
# campus-wide one. These get a shared indicator feature instead, so each
# site's model can learn its own response, including no response at all.
MINOR_HOLIDAYS = {
    'mlk_day':        lambda y: nth_weekday(y, 1, 0, 3),
    'presidents_day': lambda y: nth_weekday(y, 2, 0, 3),
    'memorial_day':   lambda y: last_monday(y, 5),
    'labor_day':      lambda y: nth_weekday(y, 9, 0, 1),
}


def _holiday_mask(dt: pd.Series, registry: dict) -> pd.Series:
    out = pd.Series(False, index=dt.index)
    if dt.empty:
        return out
    dates = dt.dt.normalize()
    for fn in registry.values():
        for y in range(dt.dt.year.min(), dt.dt.year.max() + 1):
            out = out | (dates == fn(y))
    return out


def holiday_closure_mask(dt: pd.Series) -> pd.Series:
    return _holiday_mask(dt, HARD_CLOSURES)


def minor_holiday_mask(dt: pd.Series) -> pd.Series:
    return _holiday_mask(dt, MINOR_HOLIDAYS)


def holiday_name(target) -> str | None:
    d = pd.Timestamp(target).normalize()
    for name, fn in HARD_CLOSURES.items():
        if d == fn(d.year):
            return name
    return None


def is_closed(site: str, dt: pd.Series) -> pd.Series:
    """Structural closures: Sundays, holidays, and breaks at most sites.

    Holidays were added after the test year showed a 293-parcel error on Labor
    Day 2026 - actual zero, forecast 293 - because the closure calendar knew
    only about Sundays and long breaks. Fixed holidays were being trained on as
    ordinary demand and then predicted as ordinary days.
    """
    closed = sunday_mask(dt) | holiday_closure_mask(dt)
    if site not in cfg.OPEN_ALL_YEAR:
        closed = closed | summer_mask(dt) | winter_mask(dt)
    return closed


def closure_reason(site: str, target) -> str | None:
    s = pd.Series([pd.Timestamp(target)])
    if sunday_mask(s).iloc[0]:
        return 'sunday_closed'
    if (hol := holiday_name(target)) is not None:
        return hol
    if site in cfg.OPEN_ALL_YEAR:
        return None
    if summer_mask(s).iloc[0]:
        return 'summer_break'
    if winter_mask(s).iloc[0]:
        return 'winter_break'
    return None


def classify_period(dt: pd.Series) -> pd.Series:
    """One label per date, for stratifying dwell curves.

    Order matters: move_in overlaps the tail of summer_break in the calendar
    definitions above, and move_in is the more informative label.
    """
    out = pd.Series('session', index=dt.index, dtype='object')
    out[summer_mask(dt)]  = 'summer_break'
    out[winter_mask(dt)]  = 'winter_break'
    out[spring_mask(dt)]  = 'spring_break'
    out[move_out_mask(dt)] = 'move_out'
    out[move_in_mask(dt)]  = 'move_in'
    return out


# ------------------------------------------------------------- events
# Retail/social events that move parcel volume, with ASYMMETRIC windows:
# ordering ramps up before the date and tails off after it. Windows were set
# from the measured lift profile, then rounded - the profile is built from only
# two occurrences of each event, so precise per-day bounds would be fitting
# noise.
#
# Measured lift vs a same-weekday local baseline (system-wide open-day intake):
#   halloween     1.57x (2024)   2.08x (2025)   <- strongest signal in the data
#   valentines    1.44x (2025)   1.55x (2026)
#   super_bowl    1.14x (2025)   1.19x (2026)   <- weak but consistent
#   black_friday  0.52x (2024)   0.51x (2025)   <- a TROUGH, not a peak
#   cyber_monday  0.35x (2024)   0.68x (2025)   <- likewise
#   st_patricks   1.06x, 0.88x                  <- no signal, excluded
#
# Black Friday being suppressed is not an error: students leave for
# Thanksgiving and ship to home addresses, so campus intake falls exactly when
# national retail peaks. The feature is kept so the model can learn the
# negative coefficient rather than being surprised by it every November.
EVENTS = [
    {'name': 'halloween',    'lead': 10, 'lag': 5,
     'fn': lambda y: pd.Timestamp(y, 10, 31)},
    {'name': 'valentines',   'lead': 7,  'lag': 3,
     'fn': lambda y: pd.Timestamp(y, 2, 14)},
    {'name': 'super_bowl',   'lead': 5,  'lag': 2,
     'fn': lambda y: nth_weekday(y, 2, 6, 1) + pd.Timedelta(days=1)},
    {'name': 'black_friday', 'lead': 3,  'lag': 5,
     'fn': lambda y: nth_weekday(y, 11, 3, 4) + pd.Timedelta(days=1)},
]


def event_ramp(dt: pd.Series, anchor_fn, lead: int, lag: int) -> pd.Series:
    """Triangular intensity peaking at the event, 0 outside the window.

    One coefficient per event instead of one per day in the window. With two
    observations of each event in two years of data, a per-day dummy set would
    have more parameters than evidence; a ramp asks the model a single
    question - how much does proximity to this date move volume - and lets the
    asymmetry of lead vs lag carry the ordering behaviour.
    """
    out = pd.Series(0.0, index=dt.index)
    for y in range(dt.dt.year.min() - 1, dt.dt.year.max() + 2):
        offset = (dt - anchor_fn(y)).dt.days
        before = (offset < 0) & (offset >= -lead)
        after = (offset >= 0) & (offset <= lag)
        out[before] = np.maximum(out[before], 1.0 + offset[before] / lead)
        out[after] = np.maximum(out[after], 1.0 - offset[after] / max(lag, 1))
    return out


def add_features(df: pd.DataFrame, t0: pd.Timestamp, tspan: int,
                 target: str = 'intake') -> pd.DataFrame:
    """Calendar + momentum features for one site's daily series."""
    d, dt = df.copy(), df['date']
    d['t_scaled'] = (dt - t0).dt.days / max(tspan, 1)

    dow = dt.dt.dayofweek
    d['dow_sin'] = np.sin(2 * np.pi * dow / 7)
    d['dow_cos'] = np.cos(2 * np.pi * dow / 7)
    d['is_weekend']  = (dow >= 5).astype(int)
    d['is_saturday'] = (dow == 5).astype(int)
    # Saturday hours are being cut over time; let the model see the trend.
    d['sat_trend'] = d['is_saturday'] * d['t_scaled']
    # A Saturday during a break is a different animal from a Saturday in
    # session. At Tower B, 31 of 103 training Saturdays fall in summer and
    # average 3.3 parcels against 51.2 in session, which dragged the learned
    # Saturday level down 28% and pushed three of the last four test Saturdays
    # outside the q80 band. Not modelled as a closure: 11 of those 31 days
    # have real activity, so forcing zero would be wrong.
    d['saturday_in_break'] = (
        d['is_saturday'] * (summer_mask(dt) | winter_mask(dt)).astype(int))
    d['is_minor_holiday'] = minor_holiday_mask(dt).astype(int)

    d['summer_break'] = summer_mask(dt).astype(int)
    d['winter_break'] = winter_mask(dt).astype(int)
    d['spring_break'] = spring_mask(dt).astype(int)
    d['move_in']      = move_in_mask(dt).astype(int)
    d['move_out']     = move_out_mask(dt).astype(int)

    for name, fn in (('christmas',    lambda y: pd.Timestamp(y, 12, 25)),
                     ('thanksgiving', lambda y: nth_weekday(y, 11, 3, 4))):
        near   = pd.Series(np.inf, index=d.index)
        signed = pd.Series(0.0, index=d.index)
        for y in range(dt.dt.year.min() - 1, dt.dt.year.max() + 2):
            delta = (dt - fn(y)).dt.days
            near = np.minimum(near, delta.abs())
            hit = delta.abs() <= cfg.EVENT_PAD
            signed[hit] = delta[hit]
        d[f'ev_{name}']      = (near <= cfg.EVENT_PAD).astype(int)
        d[f'ev_{name}_dist'] = signed / cfg.EVENT_PAD

    for ev in EVENTS:
        d[f"ev_{ev['name']}"] = event_ramp(dt, ev['fn'], ev['lead'], ev['lag'])

    for w in (7, 14, 28):
        d[f'roll{w}'] = (d[target].shift(1)
                         .rolling(w, min_periods=max(2, w // 2)).mean())
    return d


EVENT_FEATURES = [f"ev_{e['name']}" for e in EVENTS]

CAL = ['t_scaled', 'dow_sin', 'dow_cos', 'is_weekend', 'is_saturday', 'sat_trend',
       'saturday_in_break', 'is_minor_holiday',
       'summer_break', 'winter_break', 'spring_break', 'move_in', 'move_out',
       'ev_christmas', 'ev_christmas_dist', 'ev_thanksgiving',
       'ev_thanksgiving_dist'] + EVENT_FEATURES
MOM = CAL + ['roll7', 'roll14', 'roll28']

# Lean sets drop the time index and the weaker events, keeping only the two
# with replicated, strong lift.
# Lean sets carry saturday_in_break (the evidence for it is unambiguous) but
# NOT is_minor_holiday, so cross-validation has a variant without it and can
# reject the minor-holiday signal per site if it does not help.
LEAN_CAL = ['dow_sin', 'dow_cos', 'is_saturday', 'sat_trend',
            'saturday_in_break', 'summer_break',
            'winter_break', 'spring_break', 'move_in', 'move_out',
            'ev_halloween', 'ev_valentines']
LEAN_MOM = LEAN_CAL + ['roll14', 'roll28']

# Event-free variants. Leave-one-event-out testing showed the ramps cut MAE by
# 19-23% at Tower B and 11-19% at Sutherland, but made Bouquet Gardens
# materially worse. Rather than impose them everywhere, both variants go into
# the candidate pool and cross-validation picks per site.
NOEV_CAL = [f for f in CAL if f not in EVENT_FEATURES]
NOEV_MOM = [f for f in MOM if f not in EVENT_FEATURES]
LEAN_NOEV_CAL = [f for f in LEAN_CAL
                 if f not in ('ev_halloween', 'ev_valentines')]
LEAN_NOEV_MOM = LEAN_NOEV_CAL + ['roll14', 'roll28']


def classify_regime(dt: pd.Series) -> pd.Series:
    """Label each date with its dwell regime (academic year, Aug 1 boundary)."""
    out = pd.Series('unknown', index=dt.index, dtype='object')
    for label, start, end in cfg.REGIME_BOUNDARIES:
        hit = (dt >= pd.Timestamp(start)) & (dt <= pd.Timestamp(end))
        out[hit] = label
    return out