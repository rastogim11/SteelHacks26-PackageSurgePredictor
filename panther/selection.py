"""
Intake forecasting: candidate pool, honest selection, quantile outputs.

WHAT WAS WRONG IN v6
v6 ranked ~12 candidates by test-set MAE and then reported that same test MAE
as the model's performance. With 12 candidates x 2 horizons x 7 sites, the
published figures (74.2% improvement at Tower B) were optimistically biased by
an unknown amount. Selection and evaluation cannot share a split.

WHAT HAPPENS HERE
  1. CHRONOLOGICAL 80/20 split. The last 20% of days is a holdout that is
     never used for fitting or selection. Chronological, not random: a random
     split would let the model train on next Tuesday to predict last Monday.
  2. ROLLING-ORIGIN CV inside the 80%. Expanding-window folds with a gap equal
     to the forecast horizon, so short-horizon and long-horizon models are each
     selected under the conditions they will actually face.
  3. Winner is scored ONCE on the holdout. That number is what gets reported.

METRICS
  Primary is PINBALL LOSS AT q80, not MAE. Staffing loss is asymmetric:
  under-staffing means queues and overtime, over-staffing means mild idle
  time. MAE selects the model that is best at the middle of the distribution,
  which is not what the roster is bought against.

  MASE (seasonal-naive denominator, m=7) is reported for comparability across
  sites whose volumes differ by 70x. v6's baseline was a train-mean, which is
  a very weak bar for weekly-seasonal data.

  R2 is computed but deliberately not used for selection. At single-digit
  daily counts it is dominated by a few spikes and ranks models misleadingly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import BayesianRidge, PoissonRegressor, TweedieRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from . import calendar_features as cal
from . import config as cfg
from . import data as dat


# ------------------------------------------------------------------ metrics
def pinball(y, yhat, q: float) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    delta = y - yhat
    return float(np.mean(np.maximum(q * delta, (q - 1) * delta)))


def mase(y_true, y_pred, y_train, m: int = 7) -> float:
    """Mean absolute scaled error against a seasonal-naive benchmark."""
    y_train = np.asarray(y_train, float)
    if len(y_train) <= m:
        return float('nan')
    scale = np.mean(np.abs(y_train[m:] - y_train[:-m]))
    if scale <= 0:
        return float('nan')
    return float(mean_absolute_error(y_true, y_pred) / scale)


def nb_quantile(mu: np.ndarray, phi: float, q: float) -> np.ndarray:
    """Quantile of a count distribution with variance phi * mu.

    Quasi-Poisson dispersion is mapped onto a negative binomial so the
    interval widens with volume instead of applying one flat sigma to a
    3-parcel day and a 300-parcel day alike. phi <= 1 degenerates to Poisson.
    """
    mu = np.clip(np.asarray(mu, float), 1e-6, None)
    if phi <= 1.01:
        return stats.poisson.ppf(q, mu)
    r = mu / (phi - 1.0)
    p = r / (r + mu)
    return stats.nbinom.ppf(q, r, p)


# --------------------------------------------------------------- candidates
# Model factories are module-level named functions, not lambdas, so that
# fitted specs can be pickled into the nightly model cache.
def _make_poisson():
    return PoissonRegressor(alpha=1e-4, max_iter=3000)


def _make_poisson_l2():
    return PoissonRegressor(alpha=1.0, max_iter=3000)


def _make_tweedie():
    return TweedieRegressor(power=1.5, alpha=0.5, max_iter=3000)


def _make_bayes_ridge():
    return BayesianRidge()


def _make_hgb_poisson():
    return HistGradientBoostingRegressor(
        loss='poisson', max_depth=3, max_iter=200,
        learning_rate=0.05, min_samples_leaf=20, random_state=0)


def _make_hgb_quantile(q):
    return HistGradientBoostingRegressor(
        loss='quantile', quantile=q, max_depth=3, max_iter=200,
        learning_rate=0.05, min_samples_leaf=20, random_state=0)


# Parsimony ranking, used to break near-ties. Lower is simpler. Gradient
# boosting sits last because it has by far the most capacity to fit noise, and
# at 192 training days that capacity is a liability rather than an asset.
COMPLEXITY = {
    'bayes_ridge': 0,
    'poisson_l2': 1,
    'poisson': 2,
    'tweedie': 3,
    'hgb_poisson': 4,
}


def candidate_specs() -> list[dict]:
    """Model pool. `quantile` models predict a quantile directly; `mean`
    models predict a conditional mean and get quantiles via nb_quantile."""
    return [
        {'name': 'poisson',      'kind': 'mean',     'make': _make_poisson},
        {'name': 'poisson_l2',   'kind': 'mean',     'make': _make_poisson_l2},
        {'name': 'tweedie',      'kind': 'mean',     'make': _make_tweedie},
        {'name': 'bayes_ridge',  'kind': 'mean',     'make': _make_bayes_ridge},
        {'name': 'hgb_poisson',  'kind': 'mean',     'make': _make_hgb_poisson},
    ]
    # _make_hgb_quantile is deliberately NOT in the pool. It predicts one
    # quantile directly, so a model fitted at q80 cannot report a median, and
    # the dashboard needs both. When it won at Residences on Bigelow the whole
    # accuracy row came back NaN. Every remaining spec predicts a conditional
    # mean and derives any quantile from it via nb_quantile.


FEATURE_SETS = {
    'short': {'full': cal.MOM,  'lean': cal.LEAN_MOM,
              'full_noev': cal.NOEV_MOM, 'lean_noev': cal.LEAN_NOEV_MOM},
    'long':  {'full': cal.CAL,  'lean': cal.LEAN_CAL,
              'full_noev': cal.NOEV_CAL, 'lean_noev': cal.LEAN_NOEV_CAL},
}


def build_site_frame(tx: pd.DataFrame, site: str) -> pd.DataFrame:
    """Daily open-day series for one site, with features attached.

    Closed days are dropped rather than modelled. They are structural zeros -
    the mailroom was shut - so training on them teaches the model to predict
    zero on days nobody was ever going to deliver, and forecasting them is a
    rule, not a regression.
    """
    panel = dat.daily_panel(tx, site)
    if panel.empty:
        return panel
    panel['closed'] = cal.is_closed(site, panel['date']).to_numpy()
    open_days = panel[~panel['closed']].reset_index(drop=True)
    t0 = panel['date'].min()
    tspan = max((panel['date'].max() - t0).days, 1)
    feat = cal.add_features(open_days, t0, tspan, target='intake')
    return feat.dropna(subset=['roll28']).reset_index(drop=True)


# ----------------------------------------------------------------- folding
def rolling_origin_folds(n: int, n_folds: int, min_train: int,
                         horizon: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Expanding-window folds with a forecast gap.

    The gap matters: without it a 'short horizon' model is validated on the day
    immediately after its training data, which is easier than the 14-day-out
    case it will be asked to handle in production.
    """
    folds = []
    if n <= min_train + horizon:
        return folds
    span = n - min_train - horizon
    step = max(span // n_folds, 1)
    for i in range(n_folds):
        train_end = min_train + i * step
        test_start = train_end + horizon
        test_end = min(test_start + step, n)
        if test_start >= n or test_end <= test_start:
            break
        folds.append((np.arange(0, train_end), np.arange(test_start, test_end)))
    return folds


def _clamp_trend(Xtr: np.ndarray, Xte: np.ndarray,
                 feats: list[str]) -> np.ndarray:
    """Cap the time-index feature at its training maximum before predicting.

    t_scaled is a linear time index. Outside the fitted range a linear model
    keeps extrapolating the trend, and over a six-month holdout that walked
    Nordenberg's forecast down to 0.0 parcels/day against an actual 5.4 - the
    model had learned a mild downward drift and then ran it off the end of the
    data. Clamping converts the trend into a level once prediction leaves the
    observed range, which is the conservative reading: "we have no evidence
    the drift continues".

    Rolling-window CV cannot catch this, because its folds never extrapolate
    far beyond their own training data. Only the held-out tail exposes it.
    """
    if 't_scaled' not in feats:
        return Xte
    j = feats.index('t_scaled')
    out = Xte.copy()
    out[:, j] = np.minimum(out[:, j], Xtr[:, j].max())
    return out


def live_features(d: pd.DataFrame, feats: list[str],
                  tr: np.ndarray | None = None) -> list[str]:
    """Drop features that never vary inside the training data.

    A constant column carries no information but is not harmless: tree models
    will still split on it given a small training set, and it dilutes the
    scaler. Several columns are structurally constant at particular sites -
    summer_break and winter_break are always zero wherever those days are
    dropped as closures, and Residences on Bigelow's training window opens in
    April 2025 so it never sees winter, spring break, Halloween or
    Valentine's, leaving 10 dead columns out of the full set.

    Leaving them in let hgb_poisson overfit Sutherland to a +23 parcel bias.
    """
    sub = d.iloc[tr] if tr is not None else d
    return [f for f in feats if f in sub.columns and sub[f].nunique() > 1]


def _fit_predict(spec: dict, feats: list[str], d: pd.DataFrame,
                 tr: np.ndarray, te: np.ndarray,
                 quantile: float) -> tuple[np.ndarray, float]:
    feats = live_features(d, feats, tr) or feats
    X = d[feats].to_numpy()
    y = d['intake'].to_numpy()
    X_te_capped = _clamp_trend(X[tr], X[te], feats)
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xte = scaler.transform(X[tr]), scaler.transform(X_te_capped)

    if spec['kind'] == 'quantile':
        model = spec['make'](quantile).fit(Xtr, y[tr])
        return np.clip(model.predict(Xte), 0, None), float('nan')

    model = spec['make']().fit(Xtr, y[tr])
    mu_tr = np.clip(model.predict(Xtr), 1e-6, None)
    phi = float(np.mean((y[tr] - mu_tr) ** 2 / np.clip(mu_tr, 1, None)))
    mu_te = np.clip(model.predict(Xte), 0, None)
    return nb_quantile(mu_te, phi, quantile), phi


def select_for_site(tx: pd.DataFrame, site: str,
                    quantile: float | None = None,
                    split_date: str | None = None) -> dict:
    """Select a model for one site using year one only.

    Cross-validation runs entirely inside the training year. The test year is
    never touched here - it is scored separately by the `test` command, once.
    """
    quantile = quantile if quantile is not None else cfg.STAFFING_QUANTILE
    split_date = pd.Timestamp(split_date or cfg.SPLIT_DATE)
    d = build_site_frame(tx, site)
    result = {'site': site, 'n_open_days': len(d), 'leaderboard': [], 'chosen': {}}
    if d.empty:
        result['status'] = 'no_data'
        return result

    is_train = (d['date'] < split_date).to_numpy()
    train, holdout = np.where(is_train)[0], np.where(~is_train)[0]
    result['n_train_days'] = int(len(train))
    result['n_test_days'] = int(len(holdout))
    result['split_date'] = str(split_date.date())
    if len(train) < cfg.MIN_TRAIN_DAYS:
        result['status'] = 'insufficient_train_history'
        return result
    split = len(train)

    for mode, horizon in (('short', 1), ('long', cfg.SHORT_HORIZON + 1)):
        folds = rolling_origin_folds(
            split, cfg.CV_FOLDS,
            min_train=min(cfg.MIN_TRAIN_DAYS, max(split - horizon - 20, 30)),
            horizon=horizon)
        if not folds:
            continue

        scored = []
        for spec in candidate_specs():
            for fname, feats in FEATURE_SETS[mode].items():
                losses, maes = [], []
                for tr, te in folds:
                    try:
                        pred, _ = _fit_predict(spec, feats, d, tr, te, quantile)
                    except Exception:
                        losses = []
                        break
                    y = d['intake'].to_numpy()[te]
                    losses.append(pinball(y, pred, quantile))
                    maes.append(mean_absolute_error(y, pred))
                if losses:
                    scored.append({
                        'model': f"{spec['name']}|{fname}",
                        'spec': spec, 'feats': feats,
                        'fold_losses': list(losses),
                        'cv_pinball': float(np.mean(losses)),
                        'cv_mae': float(np.mean(maes)),
                    })
        if not scored:
            continue
        scored.sort(key=lambda r: r['cv_pinball'])

        # PARSIMONY TIE-BREAK.
        # Taking the outright lowest CV score chose hgb_poisson for Sutherland
        # by a margin of about 0.05, then missed the test year by +23 parcels a
        # day - worse than predicting last week. Boosted trees fit 192 training
        # days closely enough that five folds cannot separate skill from
        # memorisation.
        #
        # A one-standard-error rule was tried first and was far too wide: the
        # folds span different seasons, so the spread is large, 12 to 20
        # candidates landed inside 1 SE, and every site collapsed onto the same
        # simplest model - discarding a genuine 17% edge at Tower B.
        #
        # A relative tolerance behaves properly: candidates within
        # SELECTION_TOLERANCE of the best count as tied, and among those the
        # simpler model wins. Real differences survive; decimal places do not.
        top = scored[0]
        threshold = top['cv_pinball'] * (1.0 + cfg.SELECTION_TOLERANCE)
        contenders = [r for r in scored if r['cv_pinball'] <= threshold]
        best = min(contenders, key=lambda r: (
            COMPLEXITY.get(r['model'].split('|')[0], 9), len(r['feats'])))
        result.setdefault('tie_break', {})[mode] = {
            'outright_best': top['model'],
            'outright_cv': round(top['cv_pinball'], 3),
            'tolerance': cfg.SELECTION_TOLERANCE,
            'n_tied': len(contenders),
            'selected': best['model'],
            'selected_cv': round(best['cv_pinball'], 3),
        }

        # Single, final touch of the holdout.
        pred, phi = _fit_predict(best['spec'], best['feats'], d,
                                 train, holdout, quantile)
        y_hold = d['intake'].to_numpy()[holdout]
        y_train = d['intake'].to_numpy()[train]
        seasonal_naive = d['intake'].shift(7).to_numpy()[holdout]
        ok = ~np.isnan(seasonal_naive)

        # MAE and MASE on a q80 forecast are inflated by construction - the
        # forecast is meant to sit above the actual most of the time. The
        # median forecast is what belongs next to a seasonal-naive benchmark,
        # so report both and keep pinball@q80 as the selection metric.
        if best['spec']['kind'] == 'mean':
            pred_med, _ = _fit_predict(best['spec'], best['feats'], d,
                                       train, holdout, 0.50)
            mae_med = round(float(mean_absolute_error(y_hold, pred_med)), 2)
            mase_med = round(mase(y_hold, pred_med, y_train), 3)
            cover = round(float(np.mean(y_hold <= pred)), 3)
        else:
            mae_med = mase_med = None
            cover = round(float(np.mean(y_hold <= pred)), 3)

        result['chosen'][mode] = {
            'model': best['model'], 'spec': best['spec'], 'feats': best['feats'],
            'phi': phi, 'quantile': quantile,
            'cv_pinball': round(best['cv_pinball'], 3),
            'holdout_pinball': round(pinball(y_hold, pred, quantile), 3),
            'holdout_mae_q80': round(float(mean_absolute_error(y_hold, pred)), 2),
            'holdout_mae_q50': mae_med,
            'holdout_mase_q50': mase_med,
            # Share of holdout days the q80 forecast actually covered. Should
            # land near 0.80; far off means the dispersion estimate is wrong.
            'q80_coverage': cover,
            'naive_mae': round(float(mean_absolute_error(
                y_hold[ok], seasonal_naive[ok])), 2) if ok.any() else None,
            'n_folds': len(folds),
        }
        for r in scored:
            result['leaderboard'].append({
                'site': site, 'mode': mode, 'model': r['model'],
                'cv_pinball': round(r['cv_pinball'], 3),
                'cv_mae': round(r['cv_mae'], 2),
                'selected': r['model'] == best['model'],
            })

    result['status'] = 'ok' if result['chosen'] else 'no_valid_folds'
    return result


def fit_final(tx: pd.DataFrame, site: str, chosen: dict,
              train_only: bool = False):
    """Refit the selected model on ALL available data for forward use.

    Selection is done; the holdout has served its purpose. Withholding the most
    recent 20% of days from the production model would throw away the freshest
    signal for no remaining benefit.
    """
    d = build_site_frame(tx, site)
    if train_only:
        d = d[d['date'] < pd.Timestamp(cfg.SPLIT_DATE)].reset_index(drop=True)
    feats = live_features(d, chosen['feats']) or chosen['feats']
    X = d[feats].to_numpy()
    y = d['intake'].to_numpy()
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    spec = chosen['spec']
    if spec['kind'] == 'quantile':
        model = spec['make'](chosen['quantile']).fit(Xs, y)
        phi = None
    else:
        model = spec['make']().fit(Xs, y)
        mu = np.clip(model.predict(Xs), 1e-6, None)
        phi = float(np.mean((y - mu) ** 2 / np.clip(mu, 1, None)))
    return {'model': model, 'scaler': scaler, 'spec': spec,
            'feats': feats, 'phi': phi,
            't_scaled_cap': float(d['t_scaled'].max()),
            'dow_ceiling': dow_ceiling(d),
            'quantile': chosen['quantile'], 'frame': d,
            't0': d['date'].min(),
            'tspan': max((d['date'].max() - d['date'].min()).days, 1)}


def dow_ceiling(d: pd.DataFrame, headroom: float = 1.25) -> dict[int, float]:
    """Highest plausible intake per day-of-week, from observed history.

    Guards against a failure mode of multiplicative (log-link) models: a
    strong event coefficient multiplies against a low-volume weekday
    coefficient with nothing to stop it. Halloween 2026 falls on a Saturday,
    and because the event landed on a Thursday in 2024 and a Friday in 2025
    the model had never seen that combination - it forecast 355 parcels for
    Tower B against an all-time session-Saturday maximum of 175.

    The ceiling is the observed max for that weekday plus headroom, so genuine
    records can still be exceeded a little, but physically implausible values
    cannot be published. Deliberately arithmetic rather than learned: it is a
    sanity bound, and a bound that can itself be wrong is not a bound.
    """
    out = {}
    dow = d['date'].dt.dayofweek
    for w in range(7):
        vals = d.loc[dow == w, 'intake']
        if len(vals):
            out[w] = float(vals.max() * headroom)
    return out


def _apply_ceiling(pred: np.ndarray, frame: pd.DataFrame,
                   fitted: dict) -> np.ndarray:
    """Clip predictions to the observed day-of-week ceiling, where known."""
    caps = fitted.get('dow_ceiling')
    if not caps or 'date' not in frame.columns:
        return pred
    limits = pd.to_datetime(frame['date']).dt.dayofweek.map(caps).to_numpy(float)
    limits = np.where(np.isnan(limits), np.inf, limits)
    return np.minimum(np.asarray(pred, float), limits)


def predict_quantile(fitted: dict, frame: pd.DataFrame, q: float) -> np.ndarray:
    """Predict a given quantile from a fitted model on a featured frame."""
    X = frame[fitted['feats']].fillna(0).to_numpy()
    if 't_scaled' in fitted['feats'] and fitted.get('t_scaled_cap') is not None:
        j = fitted['feats'].index('t_scaled')
        X[:, j] = np.minimum(X[:, j], fitted['t_scaled_cap'])
    Xs = fitted['scaler'].transform(X)
    if fitted['spec']['kind'] == 'quantile':
        # Trained at one quantile; other quantiles are not available from it.
        raw = np.clip(fitted['model'].predict(Xs), 0, None)
    else:
        mu = np.clip(fitted['model'].predict(Xs), 0, None)
        raw = nb_quantile(mu, fitted['phi'] or 1.0, q)
    return _apply_ceiling(raw, frame, fitted)


def backtest_site(tx: pd.DataFrame, site: str, mode: str = 'short',
                  quantile: float | None = None) -> pd.DataFrame:
    """Predicted vs actual on the untouched 20% holdout, day by day.

    The model is selected by CV inside the 80% and fitted on the 80% ONLY, so
    every row here is a genuine out-of-sample prediction. This is the table to
    look at when judging accuracy - not `run week --as-of`, which uses a model
    fitted on all data including the week being 'predicted'.
    """
    quantile = quantile if quantile is not None else cfg.STAFFING_QUANTILE
    res = select_for_site(tx, site, quantile)
    if res['status'] != 'ok' or mode not in res['chosen']:
        return pd.DataFrame()

    chosen = res['chosen'][mode]
    d = build_site_frame(tx, site)
    split = int(len(d) * 0.80)
    train, holdout = np.arange(split), np.arange(split, len(d))

    pred_q80, _ = _fit_predict(chosen['spec'], chosen['feats'], d,
                               train, holdout, quantile)
    if chosen['spec']['kind'] == 'mean':
        pred_q50, _ = _fit_predict(chosen['spec'], chosen['feats'], d,
                                   train, holdout, 0.50)
    else:
        pred_q50 = np.full(len(holdout), np.nan)

    actual = d['intake'].to_numpy()[holdout]
    naive = d['intake'].shift(7).to_numpy()[holdout]

    out = pd.DataFrame({
        'site': site,
        'mode': mode,
        'model': chosen['model'],
        'date': d['date'].to_numpy()[holdout],
        'actual': actual,
        'pred_q50': np.round(pred_q50, 1),
        'pred_q80': np.round(pred_q80, 1),
        'seasonal_naive': naive,
    })
    out['day_name'] = pd.to_datetime(out['date']).dt.day_name()
    out['error_q50'] = (out['pred_q50'] - out['actual']).round(1)
    out['abs_error_q50'] = out['error_q50'].abs()
    out['covered_by_q80'] = out['actual'] <= out['pred_q80']
    return out


def event_holdout_test(tx: pd.DataFrame, site: str, event: dict,
                       year: int) -> dict | None:
    """Leave-one-event-out check: does the event ramp earn its place?

    Training stops two weeks BEFORE the ramp begins, so the model has never
    seen this occurrence. Predicting the window with and without the ramp
    isolates the feature's contribution on exactly the days it claims to
    explain. The ordinary 80/20 holdout cannot do this - it spans March to
    September and contains neither Halloween nor Valentine's.
    """
    anchor = event['fn'](year)
    lead, lag = event['lead'], event['lag']
    feat = f"ev_{event['name']}"

    d = build_site_frame(tx, site)
    if d.empty or feat not in d.columns:
        return None
    win = ((d['date'] >= anchor - pd.Timedelta(days=lead))
           & (d['date'] <= anchor + pd.Timedelta(days=lag))).to_numpy()
    tr = (d['date'] < anchor - pd.Timedelta(days=lead + 14)).to_numpy()
    if win.sum() < 5 or tr.sum() < 150:
        return None

    y = d['intake'].to_numpy()
    # Base set deliberately excludes ALL event ramps, then the one under test
    # is added back. Comparing against LEAN_CAL would be comparing a set with
    # itself for any event LEAN_CAL does not already carry.
    base = [f for f in cal.LEAN_CAL if f not in cal.EVENT_FEATURES]
    scores = {}
    for label, feats in (('with', base + [feat]), ('without', base)):
        X = d[feats].to_numpy()
        scaler = StandardScaler().fit(X[tr])
        model = _make_poisson().fit(scaler.transform(X[tr]), y[tr])
        pred = np.clip(model.predict(scaler.transform(X[win])), 0, None)
        scores[label] = float(mean_absolute_error(y[win], pred))

    return {
        'site': site, 'event': event['name'], 'year': year,
        'window_days': int(win.sum()),
        'actual_mean': round(float(y[win].mean()), 1),
        'mae_with_event': round(scores['with'], 2),
        'mae_without_event': round(scores['without'], 2),
        'mae_reduction_pct': round((1 - scores['with'] / scores['without']) * 100, 1),
    }


def score_test_year(tx: pd.DataFrame, site: str, chosen: dict,
                    split_date: str | None = None) -> pd.DataFrame:
    """Predict every open day of the test year, day by day.

    OPERATING MODE. The model is fitted on the training year only, but the
    rolling-average features for each test day are built from ACTUAL arrivals
    up to the previous day. That is not leakage - yesterday's count is known
    when you plan today - and it is exactly how the nightly job will run. The
    alternative, feeding the model its own predictions for a whole year,
    measures a scenario nobody operates in and diverges badly besides.
    """
    split_date = pd.Timestamp(split_date or cfg.SPLIT_DATE)
    d = build_site_frame(tx, site)
    if d.empty:
        return pd.DataFrame()
    is_train = (d['date'] < split_date).to_numpy()
    train, test = np.where(is_train)[0], np.where(~is_train)[0]
    if len(train) < cfg.MIN_TRAIN_DAYS or len(test) == 0:
        return pd.DataFrame()

    pred_q80, _ = _fit_predict(chosen['spec'], chosen['feats'], d,
                               train, test, chosen['quantile'])
    pred_q50, _ = _fit_predict(chosen['spec'], chosen['feats'], d,
                               train, test, 0.50)

    actual = d['intake'].to_numpy()[test]
    out = pd.DataFrame({
        'site': site,
        'model': chosen['model'],
        'date': d['date'].to_numpy()[test],
        'actual': actual,
        'predicted': np.round(pred_q50, 1),
        'predicted_q80': np.round(pred_q80, 1),
        'seasonal_naive': d['intake'].shift(7).to_numpy()[test],
    })
    out['day_name'] = pd.to_datetime(out['date']).dt.day_name()
    out['error'] = (out['predicted'] - out['actual']).round(1)
    out['abs_error'] = out['error'].abs()
    out['covered_by_q80'] = out['actual'] <= out['predicted_q80']
    return out


def accuracy_table(test_preds: pd.DataFrame) -> pd.DataFrame:
    """One accuracy row per mailroom.

    accuracy_pct = 100 * (1 - MAE / mean actual). Read it as "predictions land
    within this percentage of the typical day's volume." Deliberately not
    MAPE: a third of open days at the quiet sites are zero-intake, and MAPE
    divides by zero on every one of them.

    vs_naive_pct compares against predicting last week's same weekday, which
    is the bar a forecast has to clear to be worth running at all.
    """
    rows = []
    for site, g in test_preds.groupby('site', sort=False):
        mean_actual = g['actual'].mean()
        mae = g['abs_error'].mean()
        naive_mae = (g['seasonal_naive'] - g['actual']).abs().mean()
        rows.append({
            'mailroom': site,
            'model': g['model'].iloc[0],
            'test_days': len(g),
            'mean_actual': round(mean_actual, 1),
            'mean_predicted': round(g['predicted'].mean(), 1),
            'mae': round(mae, 2),
            'accuracy_pct': round(max(0.0, 100 * (1 - mae / max(mean_actual, 1e-9))), 1),
            'naive_mae': round(naive_mae, 2),
            'vs_naive_pct': round(100 * (1 - mae / max(naive_mae, 1e-9)), 1),
            'bias': round(g['error'].mean(), 2),
            'q80_coverage': round(g['covered_by_q80'].mean(), 3),
        })
    return pd.DataFrame(rows).sort_values('mean_actual', ascending=False)