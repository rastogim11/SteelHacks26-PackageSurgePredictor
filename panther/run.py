"""
Panther Post - command line.

Three commands, in order:

    python -m panther.run train      fit one model per mailroom on YEAR ONE
    python -m panther.run test       score YEAR TWO, accuracy per mailroom
    python -m panther.run predict    forecast the current week

TARGET VARIABLE
    Daily parcel INTAKE per mailroom - the count of packages received at that
    location on that date. One model per mailroom; no pooled model.

    Everything else is derived from it, not separately modelled:
        occupancy  = intake convolved with the dwell survival curve
        pickups    = intake convolved with the dwell hazard
        staff      = (intake + pickups + disposals) x handling time

TRAIN / TEST
    Year one (to 2025-09-18) trains. Year two (2025-09-19 on) tests, and is
    never touched during training or model selection. `predict` refits on both
    years, because once selection is settled there is no reason to withhold the
    most recent data from the model that goes live.
"""
from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings('ignore')

from . import config as cfg
from . import data as dat
from . import dwell as dw
from . import selection as sel
from . import staffing as stf
from . import weekplan as wp

OUT = Path(cfg.OUTPUT_DIR)
CACHE = OUT / 'models.pkl'


def _load(path=None) -> pd.DataFrame:
    for c in ([path] if path else [cfg.DATA_RAW, cfg.DATA_CLEANED]):
        if c and Path(c).exists():
            print(f'  data: {c}')
            return dat.load_transactions(c)
    sys.exit('No data file found. Run from the repo root, or pass --data PATH.')


def _banner(title: str):
    print(f'\n{"=" * 78}\n  {title}\n{"=" * 78}')


# --------------------------------------------------------------- 1. train
def cmd_train(args):
    _banner('TRAIN - one model per mailroom, fitted on year one')
    tx = _load(args.data)
    OUT.mkdir(parents=True, exist_ok=True)
    split = pd.Timestamp(cfg.SPLIT_DATE)
    print(f'  target: daily parcel intake per mailroom')
    print(f'  train : {tx["recv_date"].min().date()} to {(split - pd.Timedelta(days=1)).date()}')
    print(f'  test  : {split.date()} to {tx["recv_date"].max().date()} (untouched here)\n')

    fitted, rows, board = {}, [], []
    for site in dat.site_tiers(tx)['own']:
        res = sel.select_for_site(tx, site)
        if res['status'] != 'ok':
            print(f'  {site:24s} SKIPPED - {res["status"]} '
                  f'({res.get("n_train_days", 0)} year-one days)')
            continue
        board.extend(res['leaderboard'])
        per_mode = {m: sel.fit_final(tx, site, c, train_only=True)
                    for m, c in res['chosen'].items()}
        per_mode.setdefault('long', per_mode['short'])
        fitted[site] = {'models': per_mode, 'chosen': res['chosen']}
        c = res['chosen']['short']
        rows.append({'mailroom': site, 'model': c['model'],
                     'train_days': res['n_train_days'],
                     'test_days': res['n_test_days'],
                     'cv_pinball_q80': c['cv_pinball'],
                     'events_used': 'noev' not in c['model']})
        note = ''
        r = res.get('tie_break', {}).get('short')
        if r and r['selected'] != r['outright_best']:
            note = (f'  [simpler than {r["outright_best"]} '
                    f'({r["outright_cv"]:.2f} vs {r["selected_cv"]:.2f})]')
        print(f'  {site:24s} {c["model"]:22s} '
              f'train={res["n_train_days"]:3d}d  cv_pinball={c["cv_pinball"]:.2f}{note}')

    dwell_model = dw.DwellModel().fit(tx[tx['recv_date'] < split])
    with open(CACHE, 'wb') as fh:
        pickle.dump({'fitted': fitted, 'dwell': dwell_model,
                     'split': split, 'trained_on': 'year_one'}, fh)
    pd.DataFrame(rows).to_csv(OUT / 'trained_models.csv', index=False)
    pd.DataFrame(board).to_csv(OUT / 'model_leaderboard.csv', index=False)

    print(f'\n  {len(fitted)} models trained -> {CACHE}')
    print('  "events_used" false means cross-validation preferred the '
          'event-free feature set at that site.')


# ---------------------------------------------------------------- 2. test
def cmd_test(args):
    _banner('TEST - year two, never seen during training')
    tx = _load(args.data)
    if not CACHE.exists():
        sys.exit('  No trained models. Run: python -m panther.run train')
    blob = pickle.load(open(CACHE, 'rb'))

    frames = []
    for site, entry in blob['fitted'].items():
        t = sel.score_test_year(tx, site, entry['chosen']['short'],
                                split_date=str(blob['split'].date()))
        if not t.empty:
            frames.append(t)
    if not frames:
        sys.exit('  nothing to score')

    preds = pd.concat(frames, ignore_index=True)
    acc = sel.accuracy_table(preds)
    preds.to_csv(OUT / 'test_predictions.csv', index=False)
    acc.to_csv(OUT / 'test_accuracy.csv', index=False)

    print('\n  INTAKE ACCURACY PER MAILROOM')
    print(acc.to_string(index=False))
    print('\n  accuracy_pct  = 100 x (1 - MAE / mean actual)')
    print('  vs_naive_pct  = improvement over predicting last week, same weekday')
    print('  q80_coverage  = share of days the q80 forecast covered; target 0.80')

    dval = dw.validate(blob['dwell'], tx, list(blob['fitted']))
    dval.to_csv(OUT / 'test_occupancy_accuracy.csv', index=False)
    print('\n  OCCUPANCY ACCURACY (dwell model, parcels held)')
    print(dval[['site', 'occ_actual_mean', 'occ_mae', 'occ_mae_pct',
                'occ_bias']].to_string(index=False))

    if args.site:
        g = preds[preds['site'] == args.site]
        if g.empty:
            sys.exit(f'  no test rows for {args.site}')
        print(f'\n  DAY BY DAY - {args.site} (last {args.tail})')
        print(g[['date', 'day_name', 'actual', 'predicted', 'predicted_q80',
                 'error', 'covered_by_q80']].tail(args.tail).to_string(index=False))
    print(f'\n  full series -> {OUT}/test_predictions.csv')


# ------------------------------------------------------------- 3. predict
def cmd_predict(args):
    _banner('PREDICT - current week, models refitted on both years')
    tx = _load(args.data)
    OUT.mkdir(parents=True, exist_ok=True)
    if not CACHE.exists():
        sys.exit('  No trained models. Run: python -m panther.run train')
    blob = pickle.load(open(CACHE, 'rb'))

    # Refit the SELECTED models on all available data. Selection is already
    # done, so holding back year two would only starve the live model.
    fitted = {}
    for site, entry in blob['fitted'].items():
        per_mode = {m: sel.fit_final(tx, site, c)
                    for m, c in entry['chosen'].items()}
        per_mode.setdefault('long', per_mode['short'])
        fitted[site] = per_mode
    dwell_model = dw.DwellModel().fit(tx)

    as_of = pd.Timestamp(args.as_of) if args.as_of else tx['recv_date'].max()
    lockers = dat.locker_capacity(tx, list(fitted))

    plans = []
    for label, model in (('current', dwell_model),
                         ('disposal_30d', dwell_model.with_disposal(30))):
        plans.append(wp.build_week_plan(tx, fitted, model,
                                        as_of=as_of, scenario=label))
    plan = pd.concat(plans, ignore_index=True)
    plan['lockers'] = plan['site'].map(lockers)
    plan['locker_utilisation_pct'] = (
        plan['occupancy_hat'] / plan['lockers'] * 100).round(1)
    plan['current_headcount_fte'] = plan['site'].map(cfg.HEADCOUNT_FTE)
    plan.to_csv(OUT / 'week_plan.csv', index=False)

    cur = plan[plan['scenario'] == 'current']
    start, end = wp.week_bounds(as_of)
    print(f'\n  week {start.date()} to {end.date()}   (latest actual: {as_of.date()})')
    print('  intake = q80 forecast, the level staffing is planned against\n')

    for site, g in cur.groupby('site', sort=False):
        lk = lockers.get(site, 0)
        hc = cfg.HEADCOUNT_FTE.get(site)
        print(f'  {site}   lockers={lk}   current headcount={hc} FTE')
        show = g[g['is_open']][['day_name', 'source', 'intake', 'occupancy_hat',
                                'locker_utilisation_pct', 'total_hours',
                                'assign_fte']]
        print(show.to_string(index=False))
        closed = g[~g['is_open']]
        if len(closed):
            print(f'    closed: {", ".join(closed["day_name"] + " (" + closed["closed_reason"].astype(str) + ")")}')
        print()

    pc = wp.pool_check(cur)
    pc.to_csv(OUT / 'pool_check.csv', index=False)
    print('  STAFF TO ASSIGN PER DAY (from the pool, nobody is cut)')
    print(pc[['day_name', 'assigned_fte', 'pool_fte', 'shortfall_fte']]
          .to_string(index=False))

    disp = plan[plan['scenario'] == 'disposal_30d']
    print('\n  SCENARIO COMPARISON - peak parcels held this week')
    cmp_ = pd.DataFrame({
        'mailroom': cur.groupby('site')['occupancy_hat'].max().index,
        'current': cur.groupby('site')['occupancy_hat'].max().values,
        'with_30d_disposal': disp.groupby('site')['occupancy_hat'].max().values,
        'lockers': [lockers.get(s, 0) for s in cur.groupby('site')['occupancy_hat'].max().index],
    })
    cmp_['reduction_pct'] = (
        (1 - cmp_['with_30d_disposal'] / cmp_['current'].clip(lower=1)) * 100).round(1)
    print(cmp_.to_string(index=False))
    print(f'\n  -> {OUT}/week_plan.csv   (both scenarios, ready for Power BI)')


def main(argv=None):
    ap = argparse.ArgumentParser(prog='panther.run')
    ap.add_argument('--data', help='path to the package export')
    sub = ap.add_subparsers(dest='cmd', required=True)

    sub.add_parser('train', help='fit one model per mailroom on year one') \
       .set_defaults(func=cmd_train)

    t = sub.add_parser('test', help='score year two, accuracy per mailroom')
    t.add_argument('--site', help='also print a day-by-day table for one site')
    t.add_argument('--tail', type=int, default=15)
    t.set_defaults(func=cmd_test)

    p = sub.add_parser('predict', help='forecast the current week')
    p.add_argument('--as-of', help='treat this date as the latest actual')
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == '__main__':
    main()