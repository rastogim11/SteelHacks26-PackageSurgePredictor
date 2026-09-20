"""
Loading and cleaning of the WTS package export.

Differences from the v6 cleaning step, and why:

  1. OPERATOR ACCOUNTS ARE NOT IMPUTED.
     The WTS report puts the scanning user in the Mailroom column for some
     rows, so values like 'Jessica Yacko' (4,767 rows) and 'Devon Pleasant'
     (3,742) are people, not places. v6's cleaning pushed 4,443 of these into
     Sutherland, which inflated Sutherland's 2024-25 volume by ~32% while
     adding nothing to 2025-26 - every derived row falls before 2025-05.
     Because t_scaled is a model feature, v6 was fitting a fabricated ~32%
     downward trend at that site. Here these rows are relabelled 'Unassigned':
     retained for system-wide totals and labour accounting, excluded from
     per-site models.

  2. PURGE DAYS ARE FLAGGED.
     2026-07-01 shows 1,804 pickups and 2026-03-31 shows 1,054. These are
     administrative clear-outs, not students collecting parcels. Left in, they
     corrupt the dwell survival curves. Flagged here, censored in dwell.py.

  3. OCCUPANCY GROUND TRUTH IS COMPUTED.
     Both Received and Delivered are populated on every row, so true concurrent
     holdings are directly observable. v6 never used this and compared a single
     day's intake against locker capacity, which understates peak holdings at
     Tower B by roughly 5x.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg

TS_FORMAT = '%m/%d/%Y %I:%M:%S%p'

# Columns dropped on load. 'Unnamed: 1' is the recipient name.
PII_COLUMNS = ['Unnamed: 1']


def load_transactions(path: str | None = None) -> pd.DataFrame:
    """Load the raw export as a tidy transaction table.

    Returns one row per package with parsed timestamps, dwell hours, and a
    `site` column in which non-location values are folded into 'Unassigned'.
    """
    path = path or cfg.DATA_RAW
    df = pd.read_csv(path, low_memory=False) if str(path).endswith('.csv') \
        else pd.read_excel(path)

    df = df.drop(columns=[c for c in PII_COLUMNS if c in df.columns])

    df['received']  = pd.to_datetime(df['Received'],  format=TS_FORMAT, errors='coerce')
    df['delivered'] = pd.to_datetime(df['Delivered'], format=TS_FORMAT, errors='coerce')
    df = df.dropna(subset=['received']).copy()

    raw_site = df['Mailroom'].astype('string')
    not_a_place = (
        raw_site.isna()
        | raw_site.isin(cfg.OPERATOR_ACCOUNTS)
        | raw_site.isin(cfg.NON_MAILROOM_CODES)
    )
    df['site'] = raw_site.where(~not_a_place, 'Unassigned')
    df['site_is_known'] = ~not_a_place

    df['recv_date']  = df['received'].dt.normalize()
    df['deliv_date'] = df['delivered'].dt.normalize()
    df['dwell_hours'] = (df['delivered'] - df['received']).dt.total_seconds() / 3600.0

    # A handful of rows have delivered < received; treat as data error.
    bad = df['dwell_hours'] < 0
    df.loc[bad, ['delivered', 'deliv_date', 'dwell_hours']] = pd.NA

    df['is_open_record'] = df['delivered'].isna()   # still held at export time
    return df.reset_index(drop=True)


def site_tiers(tx: pd.DataFrame) -> dict[str, list[str]]:
    """Split live sites by how much history they have.

    'own'      - enough data to fit an independent model.
    'borrowed' - too thin to fit alone; forecast as a day-of-week share of the
                 anchor site instead of being dropped. Ruskin lands here: it
                 opened 2025-09-09 and has 146 days with packages. v6 admitted
                 it on a count of 205 because that figure counted calendar days
                 in its date range rather than days with activity.

    Anchor first in 'own', since hierarchical models need it fitted already.
    """
    known = tx[tx['site_is_known'] & ~tx['site'].isin(cfg.EXCLUDE)]
    counts = known.groupby('site').agg(
        n=('site', 'size'),
        days=('recv_date', 'nunique'),
    ).sort_values('n', ascending=False)

    own = counts[counts['days'] >= cfg.MIN_SITE_DAYS].index.tolist()
    borrowed = counts[counts['days'] < cfg.MIN_SITE_DAYS].index.tolist()
    if cfg.ANCHOR in own:
        own = [cfg.ANCHOR] + [s for s in own if s != cfg.ANCHOR]
    return {'own': own, 'borrowed': borrowed}


def modelled_sites(tx: pd.DataFrame) -> list[str]:
    """All live sites that receive a forecast, by either route."""
    tiers = site_tiers(tx)
    return tiers['own'] + tiers['borrowed']


def flag_purge_days(tx: pd.DataFrame) -> pd.DataFrame:
    """Identify administrative clear-out days from the pickup series.

    Keyed on the AGE of what is cleared rather than the volume cleared. A
    volume test cannot separate a bulk disposal from a post-break resumption
    rush, because both are large relative to a break-depressed baseline, and
    censoring the latter throws away the genuine pickup behaviour the dwell
    model exists to learn. Aged stock being cleared en masse is unambiguous.
    See config.PURGE_* for the observed separation.
    """
    picked = tx.dropna(subset=['deliv_date'])
    out = []
    for site, g in picked.groupby('site', observed=True):
        days = pd.date_range(g['deliv_date'].min(), g['deliv_date'].max(), freq='D')
        by_day = g.groupby('deliv_date')['dwell_hours']
        counts = by_day.size().reindex(days, fill_value=0)
        median_age = (by_day.median() / 24.0).reindex(days)
        is_purge = (
            (counts >= cfg.PURGE_MIN_ITEMS)
            & (median_age >= cfg.PURGE_MEDIAN_AGE_DAYS)
        ).fillna(False)
        out.append(pd.DataFrame({
            'site': site,
            'date': days,
            'pickups': counts.to_numpy(),
            'median_age_days': median_age.round(1).to_numpy(),
            'is_purge': is_purge.to_numpy(),
        }))
    return pd.concat(out, ignore_index=True)


def daily_panel(tx: pd.DataFrame, site: str) -> pd.DataFrame:
    """Daily intake, pickups, and true concurrent occupancy for one site.

    Occupancy is a step function: +1 at received, -1 at delivered, cumulated.
    This is observed fact, not a model output, and is the validation target for
    the dwell model in dwell.py.
    """
    g = tx[tx['site'] == site]
    if g.empty:
        return pd.DataFrame()

    days = pd.date_range(g['recv_date'].min(), g['recv_date'].max(), freq='D')
    intake  = g.groupby('recv_date').size().reindex(days, fill_value=0)
    pickups = g.dropna(subset=['deliv_date']).groupby('deliv_date').size() \
               .reindex(days, fill_value=0)
    occupancy = (intake - pickups).cumsum()

    return pd.DataFrame({
        'site': site,
        'date': days,
        'intake': intake.to_numpy(),
        'pickups': pickups.to_numpy(),
        'occupancy_eod': occupancy.to_numpy(),
    })


def all_daily_panels(tx: pd.DataFrame, sites: list[str] | None = None) -> pd.DataFrame:
    sites = sites or modelled_sites(tx)
    return pd.concat([daily_panel(tx, s) for s in sites], ignore_index=True)


def locker_capacity(tx: pd.DataFrame, sites: list[str]) -> dict[str, int]:
    """Locker counts per site.

    Uses cfg.LOCKERS_OVERRIDE when Operations supplies real figures. Until
    then, reproduces v6's volume-share split of an assumed pool so numbers
    stay comparable against the baseline. Both the pool size and the split are
    placeholders - see config.
    """
    if cfg.LOCKERS_OVERRIDE:
        return {s: int(cfg.LOCKERS_OVERRIDE[s]) for s in sites
                if s in cfg.LOCKERS_OVERRIDE}

    known = tx[tx['site'].isin(sites)]
    vols = known.groupby('site').size()
    alloc: dict[str, int] = {}
    for tier, share in (('large', cfg.LARGE_SHARE), ('small', 1 - cfg.LARGE_SHARE)):
        members = [s for s in sites
                   if (s in cfg.LARGE) == (tier == 'large')]
        if not members:
            continue
        sub = vols.reindex(members).fillna(0)
        total = max(sub.sum(), 1)
        for s in members:
            alloc[s] = int(round(cfg.TOTAL_LOCKERS * share * sub[s] / total))
    drift = cfg.TOTAL_LOCKERS - sum(alloc.values())
    if drift and alloc:
        alloc[max(alloc, key=lambda k: alloc[k])] += drift
    return alloc


def summarise(tx: pd.DataFrame) -> pd.DataFrame:
    """Per-site data inventory, including what was dropped and why."""
    rows = []
    for site, g in tx.groupby('site'):
        panel = daily_panel(tx, site)
        rows.append({
            'site': site,
            'packages': len(g),
            'first_seen': g['recv_date'].min().date(),
            'last_seen': g['recv_date'].max().date(),
            'active_days': g['recv_date'].nunique(),
            'median_dwell_h': round(g['dwell_hours'].median(), 1),
            'p90_dwell_h': round(g['dwell_hours'].quantile(0.90), 1),
            'peak_intake': int(panel['intake'].max()) if len(panel) else 0,
            'peak_occupancy': int(panel['occupancy_eod'].max()) if len(panel) else 0,
        })
    return pd.DataFrame(rows).sort_values('packages', ascending=False)