"""
Dwell-time survival model.

This is the fix for the flow-vs-stock error. v6 compared a single day's intake
against locker capacity, which treats arrivals as if they were occupancy. They
are not: the median package sits 24 hours, but 19.4% are still held after a
week, 12.3% after two weeks, and 6.5% after a month. Tower B's peak daily
intake is 741 while its peak concurrent holdings are 3,596.

One survival curve drives both deliverables:

    S(k)  = P(still held k days after arrival)   -> occupancy  (capacity case)
    h(k)  = S(k-1) - S(k)                        -> pickups    (staffing case)

    occupancy(t) = sum_k  intake(t-k) * S_p(k),  p = period(t-k)
    pickups(t)   = sum_k  intake(t-k) * h_p(k)

Curves are estimated per site and arrival period, because dwell is strongly
seasonal: median dwell is 19.8h for September arrivals against 95.5h for July.
Students leave, parcels stay.

Estimation is discrete-time Kaplan-Meier on daily bins, with two censoring
rules:

  * ADMINISTRATIVE PURGES are censored, not counted as events. 2026-07-01
    cleared 1,804 packages in a day. Treating that as 1,804 students choosing
    to collect would bias every curve toward fast pickup.
  * OPEN RECORDS (still held at export) are right-censored at the export date.

Kaplan-Meier handles both correctly, which is the reason for using it over a
plain empirical CDF of observed dwell.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import calendar_features as cal
from . import config as cfg
from . import data as dat

# Longest observed dwell is 506 days, and 2.4% of Tower B parcels sit past
# 180 days. Truncating at 180 discarded that residue and under-predicted Tower
# B occupancy by a mean of 226 parcels - the whole of the observed bias. The
# aged stock is not noise: it accumulated to ~2,700 parcels by March 2026 and
# was only released by the 2026-07-01 purge of 1,806 items.
MAX_LAG = 540
MIN_EVENTS = 40        # below this, a stratum falls back to the site pooled curve


class DwellModel:
    """Per-site, per-period discrete survival curves with graceful fallback."""

    def __init__(self, max_lag: int = MAX_LAG):
        self.max_lag = max_lag
        # Keyed (site, regime, period), with a fallback chain for thin strata.
        self.curves: dict[tuple[str, str, str], np.ndarray] = {}
        self.regime_curves: dict[tuple[str, str], np.ndarray] = {}
        self.site_curves: dict[str, np.ndarray] = {}
        self.global_curve: np.ndarray | None = None
        self.diagnostics: pd.DataFrame | None = None
        # Populated only by with_disposal(): fraction of arrivals still held
        # when the return-to-sender deadline lands, i.e. staff disposal work.
        self.disposal_mass: dict = {}
        self.disposal_deadline: int | None = None

    # ------------------------------------------------------------- fitting
    def _km(self, dwell_days: np.ndarray, event: np.ndarray) -> np.ndarray:
        """Discrete Kaplan-Meier survival, S(0..max_lag)."""
        K = self.max_lag
        surv = np.ones(K + 1)
        at_risk = len(dwell_days)
        if at_risk == 0:
            return surv

        # Bin events and censorings by day.
        d = np.clip(dwell_days.astype(int), 0, K)
        events = np.bincount(d[event.astype(bool)], minlength=K + 1)[:K + 1]
        censor = np.bincount(d[~event.astype(bool)], minlength=K + 1)[:K + 1]

        s = 1.0
        n = float(at_risk)
        for k in range(K + 1):
            if n > 0 and events[k] > 0:
                s *= (1.0 - events[k] / n)
            surv[k] = s
            n -= (events[k] + censor[k])
            if n <= 0:
                n = 0.0
        return surv

    def fit(self, tx: pd.DataFrame, purge: pd.DataFrame | None = None) -> 'DwellModel':
        tx = tx[tx['site_is_known']].copy()
        if purge is None:
            purge = dat.flag_purge_days(tx)

        purge_keys = set(
            zip(purge.loc[purge['is_purge'], 'site'],
                purge.loc[purge['is_purge'], 'date'])
        )

        export_end = tx['received'].max().normalize()
        tx['period'] = cal.classify_period(tx['recv_date'])
        tx['regime'] = cal.classify_regime(tx['recv_date'])

        # Event vs censored.
        picked_on_purge = pd.Series(
            list(zip(tx['site'], tx['deliv_date'])), index=tx.index
        ).isin(purge_keys)

        tx['event'] = tx['delivered'].notna() & ~picked_on_purge.values
        # Censored records contribute time-at-risk up to the censoring moment.
        censor_time = np.where(
            tx['delivered'].notna(),
            tx['dwell_hours'] / 24.0,
            (export_end - tx['recv_date']).dt.days,
        )
        tx['dwell_days'] = np.clip(np.nan_to_num(censor_time, nan=0.0), 0, None)

        diag = []
        for (site, regime, period), g in tx.groupby(
                ['site', 'regime', 'period'], observed=True):
            n_ev = int(g['event'].sum())
            curve = self._km(g['dwell_days'].to_numpy(), g['event'].to_numpy())
            if n_ev >= MIN_EVENTS:
                self.curves[(site, regime, period)] = curve
            diag.append({
                'site': site, 'regime': regime, 'period': period,
                'n': len(g), 'events': n_ev,
                'median_dwell_d': float(np.argmax(curve <= 0.5)) if (curve <= 0.5).any() else np.nan,
                'still_held_d7': round(float(curve[min(7, self.max_lag)]), 3),
                'still_held_d30': round(float(curve[min(30, self.max_lag)]), 3),
                'truncated_mass': round(float(curve[-1]), 3),
                'used': n_ev >= MIN_EVENTS,
            })

        for (site, regime), g in tx.groupby(['site', 'regime'], observed=True):
            if int(g['event'].sum()) >= MIN_EVENTS:
                self.regime_curves[(site, regime)] = self._km(
                    g['dwell_days'].to_numpy(), g['event'].to_numpy())
        for site, g in tx.groupby('site', observed=True):
            self.site_curves[site] = self._km(
                g['dwell_days'].to_numpy(), g['event'].to_numpy())
        self.global_curve = self._km(
            tx['dwell_days'].to_numpy(), tx['event'].to_numpy())

        self.diagnostics = pd.DataFrame(diag).sort_values(
            ['site', 'regime', 'period'])
        return self

    # ---------------------------------------------------------- retrieval
    def survival(self, site: str, period: str,
                 regime: str | None = None) -> np.ndarray:
        """Survival curve with graceful degradation.

        Chain: (site, regime, period) -> (site, regime) -> (site) -> global.
        Stratifying three ways leaves some cells thin, so each level falls back
        to the next rather than returning a curve fitted on a handful of
        parcels.
        """
        regime = regime or cfg.FORECAST_REGIME
        for key in ((site, regime, period),):
            if key in self.curves:
                return self.curves[key]
        if (site, regime) in self.regime_curves:
            return self.regime_curves[(site, regime)]
        if site in self.site_curves:
            return self.site_curves[site]
        return self.global_curve

    def hazard(self, site: str, period: str,
               regime: str | None = None) -> np.ndarray:
        s = self.survival(site, period, regime)
        h = np.empty_like(s)
        h[0] = 1.0 - s[0]
        h[1:] = s[:-1] - s[1:]
        return h

    def with_disposal(self, max_hold_days: int) -> 'DwellModel':
        """Return a copy whose curves enforce a return-to-sender deadline.

        Parcels uncollected after `max_hold_days` are removed by staff rather
        than sitting indefinitely. Survival is forced to zero past the
        deadline; the mass removed at the deadline is recorded separately,
        because disposal is not a student pickup - it is staff labour, and the
        staffing model must be charged for it.
        """
        clone = DwellModel(self.max_lag)
        clone.diagnostics = self.diagnostics
        cut = min(max_hold_days, self.max_lag)

        def truncate(curve):
            out = curve.copy()
            residual = float(out[cut])      # still held when the deadline hits
            out[cut + 1:] = 0.0
            return out, residual

        for store, src in (('curves', self.curves),
                           ('regime_curves', self.regime_curves),
                           ('site_curves', self.site_curves)):
            target = getattr(clone, store)
            for key, curve in src.items():
                new, resid = truncate(curve)
                target[key] = new
                clone.disposal_mass[key] = resid
        clone.global_curve = truncate(self.global_curve)[0]
        clone.disposal_deadline = cut
        return clone

    # ------------------------------------------------------- convolution
    def project(self, site: str, dates: pd.Series, intake: np.ndarray,
                initial_occupancy: float = 0.0,
                regime: str | None = None) -> pd.DataFrame:
        """Convolve an intake series into occupancy and pickup series.

        `intake` may be actual history (for validation) or forecast values.
        `initial_occupancy` accounts for the standing stock of parcels that
        arrived before the window opens; without it the first weeks of any
        reconstruction start from an empty mailroom and read far too low.
        """
        dates = pd.Series(pd.to_datetime(dates)).reset_index(drop=True)
        intake = np.asarray(intake, dtype=float)
        n = len(intake)
        periods = cal.classify_period(dates).to_numpy()
        # Each parcel's dwell follows the regime it arrived under. For history
        # that is the observed regime; for forecasts the caller pins one.
        regimes = (cal.classify_regime(dates).to_numpy() if regime is None
                   else np.full(n, regime, dtype=object))

        # Grouped by ARRIVAL period, then vectorised over lags: each parcel
        # carries the curve of the period it arrived in, so arrivals are split
        # by period first and each slice convolved with its own curve.
        occ = np.zeros(n)
        pick = np.zeros(n)
        kmax = min(self.max_lag, n - 1)
        strata = pd.MultiIndex.from_arrays([regimes, periods]).unique()
        for reg, period in strata:
            arrivals = np.where((regimes == reg) & (periods == period),
                                intake, 0.0)
            if not arrivals.any():
                continue
            s = self.survival(site, period, reg)
            h = self.hazard(site, period, reg)
            for k in range(kmax + 1):
                if s[k] <= 0 and h[k] <= 0:
                    continue
                src = arrivals[:n - k]
                if s[k] > 0:
                    occ[k:] += src * s[k]
                if h[k] > 0:
                    pick[k:] += src * h[k]

        # Standing stock decays away under the site's pooled curve.
        if initial_occupancy > 0:
            base = self.site_curves.get(site, self.global_curve)
            tail = np.arange(n)
            decay = np.where(tail <= self.max_lag,
                             base[np.clip(tail, 0, self.max_lag)], base[-1])
            occ += initial_occupancy * decay

        # Disposals land exactly `disposal_deadline` days after arrival.
        disp = np.zeros(n)
        if self.disposal_deadline is not None:
            k = self.disposal_deadline
            if k < n:
                for reg, period in strata:
                    arrivals = np.where((regimes == reg) & (periods == period),
                                        intake, 0.0)
                    if not arrivals.any():
                        continue
                    mass = (self.disposal_mass.get((site, reg, period))
                            or self.disposal_mass.get((site, reg))
                            or self.disposal_mass.get(site) or 0.0)
                    disp[k:] += arrivals[:n - k] * mass

        return pd.DataFrame({
            'site': site, 'date': dates,
            'intake': intake,
            'occupancy_hat': occ,
            'pickups_hat': pick,
            'disposals_hat': disp,
        })


def validate(model: DwellModel, tx: pd.DataFrame, sites: list[str],
             warmup_days: int = 60, tail_exclude_days: int = 60) -> pd.DataFrame:
    """Score reconstructed occupancy and pickups against observed truth.

    Because both Received and Delivered are populated, the true series is
    observable. Feeding ACTUAL intake through the model isolates dwell error
    from intake-forecast error - if this step is not accurate, nothing built
    on top of it can be.

    Two windows are excluded, both for structural reasons rather than to
    flatter the numbers:

      warmup_days       - the convolution has no arrival history before the
                          series opens, so early occupancy is necessarily low.
      tail_exclude_days - the WTS export contains only CLOSED records. Every
                          row has a Delivered timestamp, so parcels sitting in
                          a mailroom at export time are absent from the file
                          entirely. This drags observed occupancy artificially
                          to zero at the boundary (Tower B ends at exactly 0).
                          Scoring against it would penalise the model for being
                          right. Fixing this properly needs a currently-held
                          snapshot from Operations.
    """
    rows = []
    for site in sites:
        panel = dat.daily_panel(tx, site)
        if panel.empty:
            continue
        proj = model.project(site, panel['date'], panel['intake'].to_numpy())
        m = panel.merge(proj[['date', 'occupancy_hat', 'pickups_hat']], on='date')
        m = m.iloc[warmup_days:len(m) - tail_exclude_days]
        if m.empty:
            continue
        occ_err = m['occupancy_hat'] - m['occupancy_eod']
        pick_err = m['pickups_hat'] - m['pickups']
        denom = max(m['occupancy_eod'].mean(), 1.0)
        rows.append({
            'site': site,
            'days_scored': len(m),
            'occ_actual_mean': round(m['occupancy_eod'].mean(), 1),
            'occ_pred_mean': round(m['occupancy_hat'].mean(), 1),
            'occ_mae': round(occ_err.abs().mean(), 1),
            'occ_mae_pct': round(occ_err.abs().mean() / denom * 100, 1),
            'occ_bias': round(occ_err.mean(), 1),
            'occ_peak_actual': int(m['occupancy_eod'].max()),
            'occ_peak_pred': int(m['occupancy_hat'].max()),
            'pickups_mae': round(pick_err.abs().mean(), 2),
            'pickups_actual_mean': round(m['pickups'].mean(), 1),
        })
    return pd.DataFrame(rows)