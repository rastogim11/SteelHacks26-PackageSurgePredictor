"""
Panther Post — Package Surge Forecasting (v6)
=============================================
Per-mailroom daily volume forecasting with campus-occupancy and closure logic,
plus model-class selection appropriate to each mailroom's data density.

OPERATIONAL RULES ENCODED
  * Permanently closed sites dropped: Forbes, Darragh.
  * CLOSED SUNDAYS, every site. Sunday's share of volume fell 2.46% -> 0.18%
    between the 2024-25 and 2025-26 academic years, confirming the policy.
  * Saturday = open, limited hours. Observed at ~37-45% of a weekday at the
    larger sites; carried as an `is_saturday` feature rather than a rule.
  * SUMMER + WINTER BREAK: only Tower B operates. Windows derived from when
    non-Tower sites actually stop/resume scanning:
        summer  May 3  - Aug 16   (resumption spike Aug 17)
        winter  Dec 18 - Jan 2    (resumption spike Jan 3)
    Exception: Residences on Bigelow ran 165 pkgs over 56 days May-Aug, so it
    is treated as open year-round. CONFIRM WITH OPERATIONS.
  * Closed days are STRUCTURAL zeros, not demand: dropped from training and
    forced to 0 at prediction. This lowers headline R2 because predicting
    "0 on July 4" was previously free credit.

MODEL SELECTION
  * Horizon-aware, two models per mailroom:
      SHORT (<=14d): includes rolling-average momentum features.
      LONG  (>14d) : calendar/occupancy only. Prevents the recursive feedback
                     loop where predictions feed roll7 and the forecast locks
                     into a self-reinforcing plateau.
  * Candidate classes per mailroom, selected by TEST MAE:
      - Tweedie (power=1.5)  : zero-inflated overdispersed counts -> small sites
      - Poisson (log link)   : multiplicative shutdown during breaks
      - BayesianRidge        : dense, higher-count sites
      - HistGB (poisson loss): nonlinear, guarded by min_samples_leaf
      - Hierarchical         : predict as day-of-week share of Tower B, which
                               lets sparse sites borrow strength from the one
                               strong model (Tower B, R2 ~0.74)
      - Climatology          : day-of-week x period empirical means (reference)
  * MAE is the primary selection metric. For single-digit daily counts, RMSE
    and R2 are dominated by a few rare spikes and rank models misleadingly
    (e.g. Nordenberg: R2 -0.06 but 57% better MAE than baseline).
  * Intervals use quasi-Poisson dispersion (var = phi*mu) so width scales with
    volume instead of applying one flat sigma to every day.
"""
import pandas as pd, numpy as np, warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import BayesianRidge, PoissonRegressor, TweedieRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA          = 'packagestats_cleaned.xlsx'
EXCLUDE       = ['Forbes', 'Darragh']
LARGE         = ['Tower B', 'Sutherland']
OPEN_ALL_YEAR = ['Tower B', 'Residences on Bigelow']
ANCHOR        = 'Tower B'          # hierarchical models scale off this site
TOTAL_LOCKERS, LARGE_SHARE = 1000, 0.70
EVENT_PAD, SHORT_HORIZON   = 7, 14

# ----------------------------------------------------------------- calendar
def nth_weekday(y, m, wd, n):
    d = pd.Timestamp(y, m, 1)
    return d + pd.Timedelta(days=(wd - d.dayofweek) % 7 + 7*(n-1))

def summer_mask(dt):
    m, day = dt.dt.month, dt.dt.day
    return ((m==5)&(day>=3)) | m.isin([6,7]) | ((m==8)&(day<=16))

def winter_mask(dt):
    m, day = dt.dt.month, dt.dt.day
    return ((m==12)&(day>=18)) | ((m==1)&(day<=2))

def sunday_mask(dt):
    return dt.dt.dayofweek == 6

def is_closed(mailroom, dt):
    closed = sunday_mask(dt)
    if mailroom not in OPEN_ALL_YEAR:
        closed = closed | summer_mask(dt) | winter_mask(dt)
    return closed

def closure_reason(mailroom, target):
    s = pd.Series([target])
    if sunday_mask(s).iloc[0]:     return 'sunday_closed'
    if mailroom in OPEN_ALL_YEAR:  return None
    if summer_mask(s).iloc[0]:     return 'summer_break'
    if winter_mask(s).iloc[0]:     return 'winter_break'
    return None

def add_features(df, t0, tspan):
    d, dt = df.copy(), df['date']
    d['t_scaled'] = (dt - t0).dt.days / tspan
    dow = dt.dt.dayofweek
    d['dow_sin'], d['dow_cos'] = np.sin(2*np.pi*dow/7), np.cos(2*np.pi*dow/7)
    d['is_weekend'], d['is_saturday'] = (dow>=5).astype(int), (dow==5).astype(int)

    m, day = dt.dt.month, dt.dt.day
    d['summer_break'] = summer_mask(dt).astype(int)
    d['winter_break'] = winter_mask(dt).astype(int)
    d['spring_break'] = ((m==3)&(day>=5)&(day<=15)).astype(int)
    d['move_in']      = (((m==8)&(day>=17))|((m==9)&(day<=10))).astype(int)
    d['move_out']     = (((m==4)&(day>=22))|((m==5)&(day<=2))).astype(int)

    for name, fn in (('christmas',    lambda y: pd.Timestamp(y,12,25)),
                     ('thanksgiving', lambda y: nth_weekday(y,11,3,4))):
        near   = pd.Series(np.inf, index=d.index)
        signed = pd.Series(0.0,   index=d.index)
        for y in range(dt.dt.year.min()-1, dt.dt.year.max()+2):
            dd   = (dt - fn(y)).dt.days
            near = np.minimum(near, dd.abs())
            hit  = dd.abs() <= EVENT_PAD
            signed[hit] = dd[hit]
        d[f'ev_{name}']      = (near<=EVENT_PAD).astype(int)
        d[f'ev_{name}_dist'] = signed/EVENT_PAD

    for w in (7,14,28):
        d[f'roll{w}'] = d['count'].shift(1).rolling(w, min_periods=max(2,w//2)).mean()
    return d

CAL  = ['t_scaled','dow_sin','dow_cos','is_weekend','is_saturday',
        'summer_break','winter_break','spring_break','move_in','move_out',
        'ev_christmas','ev_christmas_dist','ev_thanksgiving','ev_thanksgiving_dist']
MOM  = CAL + ['roll7','roll14','roll28']
LEAN_CAL = ['dow_sin','dow_cos','is_saturday','summer_break','winter_break',
            'spring_break','move_in','move_out']
LEAN_MOM = LEAN_CAL + ['roll14','roll28']

def _estimators():
    return [
        (lambda: BayesianRidge(),                                             'bayes_ridge'),
        (lambda: PoissonRegressor(alpha=1e-4, max_iter=3000),                 'poisson'),
        (lambda: PoissonRegressor(alpha=1.0,  max_iter=3000),                 'poisson_l2'),
        (lambda: TweedieRegressor(power=1.5, alpha=0.5, max_iter=3000),       'tweedie'),
        (lambda: HistGradientBoostingRegressor(loss='poisson', max_depth=3,
                    max_iter=200, learning_rate=0.05, min_samples_leaf=20),   'hgb_poisson'),
    ]

def pois_dev(y, mu):
    mu = np.clip(np.asarray(mu,float), 1e-6, None)
    t  = np.where(y>0, y*np.log(np.clip(y,1e-9,None)/mu), 0.0)
    return float(np.mean(2*(t-(y-mu))))

# -------------------------------------------------------------------- load
raw = pd.read_excel(DATA)
raw['dt']   = pd.to_datetime(raw['Received'], format='%m/%d/%Y %I:%M:%S%p', errors='coerce')
raw['date'] = raw['dt'].dt.normalize()
raw = raw[~raw['Mailroom'].isin(EXCLUDE)].copy()

vols  = raw.groupby('Mailroom').size()
tiers = {m: ('large' if m in LARGE else 'small') for m in vols.index}

alloc = {}
for tier, share in (('large', LARGE_SHARE), ('small', 1-LARGE_SHARE)):
    mem = [m for m in vols.index if tiers[m]==tier]; sub = vols[mem]
    for m in mem:
        alloc[m] = int(round(TOTAL_LOCKERS*share*sub[m]/sub.sum()))
if (drift := TOTAL_LOCKERS - sum(alloc.values())):
    alloc[max(alloc, key=alloc.get)] += drift

def frame_for(mr):
    grp = raw[raw['Mailroom']==mr]
    idx = pd.date_range(grp['date'].min(), grp['date'].max(), freq='D')
    daily = (grp.groupby('date').size().reindex(idx, fill_value=0)
                .rename_axis('date').reset_index(name='count'))
    daily['closed'] = is_closed(mr, daily['date']).values
    op = daily[~daily['closed']].reset_index(drop=True)
    t0 = daily['date'].min(); tspan = max((daily['date'].max()-t0).days, 1)
    d  = add_features(op, t0, tspan).dropna(subset=['roll28']).reset_index(drop=True)
    return d, op, t0, tspan, int(daily['closed'].sum())

ANCHOR_DAILY = frame_for(ANCHOR)[0][['date','count']].rename(columns={'count':'anchor'})
ANCHOR_PRED  = None   # filled after the anchor site is fitted; predicted, not actual

def fit_pool(d, sp, mode, mr=None):
    """Fit all candidates for one horizon mode; return list of dicts sorted by MAE."""
    y, yte = d['count'].values, d['count'].values[sp:]
    fsets = ({'full': MOM, 'lean': LEAN_MOM} if mode=='short'
             else {'full': CAL, 'lean': LEAN_CAL})
    out = []
    for fname, feats in fsets.items():
        X = d[feats].values
        sc = StandardScaler().fit(X[:sp])
        Xtr, Xte = sc.transform(X[:sp]), sc.transform(X[sp:])
        for mk, lbl in _estimators():
            try:
                m = mk().fit(Xtr, y[:sp])
                p = m.predict(Xte).clip(0)
                out.append({'label':f'{lbl}|{fname}','kind':'ml','model':m,'scaler':sc,
                            'feats':feats,'pred_test':p})
            except Exception:
                pass
    # hierarchical: day-of-week share of the anchor site.
    # Shares are learned on TRAIN actuals, but the test-period anchor volume is
    # taken from the anchor MODEL's predictions -- using anchor actuals here
    # would leak information unavailable at forecast time.
    if mr != ANCHOR and ANCHOR_PRED is not None:
        mrg = d[['date','count']].merge(ANCHOR_DAILY, on='date', how='left')
        mrg['anchor'] = mrg['anchor'].fillna(mrg['anchor'].median())
        tr = mrg.iloc[:sp].copy(); tr['dow'] = tr['date'].dt.dayofweek
        dw = tr.groupby('dow').apply(lambda g: g['count'].sum()/max(g['anchor'].sum(),1))
        gs = float(tr['count'].sum()/max(tr['anchor'].sum(),1))
        te = mrg.iloc[sp:].copy()
        te['dow'] = te['date'].dt.dayofweek
        te['anchor_hat'] = te['date'].map(ANCHOR_PRED)
        te['anchor_hat'] = te['anchor_hat'].fillna(te['anchor'].median())
        out.append({'label':'hierarchical','kind':'hier','dow_share':dw.to_dict(),
                    'global_share':gs,'feats':None,
                    'pred_test':(te['anchor_hat']*te['dow'].map(dw).fillna(gs)).values})
    # climatology reference
    tr2 = d.iloc[:sp].copy()
    tr2['per'] = np.select([tr2['move_in']==1, tr2['spring_break']==1],
                           ['move_in','spring_break'],'session')
    tr2['dow'] = tr2['date'].dt.dayofweek
    tab, gm = tr2.groupby(['per','dow'])['count'].mean().to_dict(), float(tr2['count'].mean())
    te2 = d.iloc[sp:].copy()
    te2['per'] = np.select([te2['move_in']==1, te2['spring_break']==1],
                           ['move_in','spring_break'],'session')
    te2['dow'] = te2['date'].dt.dayofweek
    out.append({'label':'climatology','kind':'clim','table':tab,'grand_mean':gm,'feats':None,
                'pred_test':np.array([tab.get((p,w),gm) for p,w in zip(te2['per'],te2['dow'])])})

    for c in out:
        p = np.asarray(c['pred_test'], float)
        c['mae']  = mean_absolute_error(yte, p)
        c['rmse'] = np.sqrt(mean_squared_error(yte, p))
        c['r2']   = r2_score(yte, p)
        c['dev']  = pois_dev(yte, p)
    return sorted(out, key=lambda c: c['mae'])

MODELS, panel, summary, leaderboard = {}, [], [], []
globals().setdefault("ANCHOR_PRED", None)

ordered = [ANCHOR] + [m for m in vols.index if m != ANCHOR]
for mr in ordered:
    d, op, t0, tspan, n_closed = frame_for(mr)
    if len(d) < 150: continue
    y, sp = d['count'].values, int(len(d)*0.8)

    poolS, poolL = fit_pool(d, sp, 'short', mr), fit_pool(d, sp, 'long', mr)
    bS, bL = poolS[0], poolL[0]

    for mode, pool in (('short',poolS), ('long',poolL)):
        for c in pool:
            leaderboard.append({'mailroom':mr,'tier':tiers[mr],'mode':mode,'model':c['label'],
                                'mae':round(c['mae'],2),'rmse':round(c['rmse'],2),
                                'r2':round(c['r2'],3),'pois_dev':round(c['dev'],3)})

    def full_pred(c):
        if c['kind']=='ml':
            return c['model'].predict(c['scaler'].transform(d[c['feats']].values)).clip(0)
        if c['kind']=='hier':
            m2 = d[['date']].copy()
            m2['anchor'] = m2['date'].map(ANCHOR_PRED or {})
            fallback = ANCHOR_DAILY.set_index('date')['anchor']
            m2['anchor'] = m2['anchor'].fillna(m2['date'].map(fallback)).fillna(0.0)
            dw = m2['date'].dt.dayofweek.map(c['dow_share']).fillna(c['global_share'])
            return (m2['anchor']*dw).values.clip(0)
        per = np.select([d['move_in']==1, d['spring_break']==1],['move_in','spring_break'],'session')
        return np.array([c['table'].get((p,w),c['grand_mean'])
                         for p,w in zip(per, d['date'].dt.dayofweek)])

    pred = full_pred(bS)
    if mr == ANCHOR:                       # expose anchor predictions for hierarchical
        ANCHOR_PRED = dict(zip(d['date'], pred))
    phi  = float(np.mean((y[:sp]-pred[:sp])**2 / np.clip(pred[:sp],1,None)))
    sd   = np.sqrt(phi*np.clip(pred,1,None))
    cap  = alloc[mr]

    o = pd.DataFrame({'date':d['date'],'mailroom':mr,'tier':tiers[mr],
        'actual_count':y,'predicted_count':pred.round(1),'pred_sd':sd.round(1),
        'lower_95':(pred-1.96*sd).clip(0).round(1),'upper_95':(pred+1.96*sd).round(1),
        'split':['train']*sp+['test']*(len(d)-sp),
        'day_of_week':d['date'].dt.day_name(),'month_name':d['date'].dt.month_name(),
        'lockers':cap,'is_open':True,'model_used':bS['label']})
    o['utilization_pct'] = (o['predicted_count']/cap*100).round(1)
    o['overflow_units']  = (o['actual_count']-cap).clip(lower=0)
    o['risk_flag'] = np.select(
        [o['upper_95']>cap, o['predicted_count']>cap*0.8, o['predicted_count']>cap*0.5],
        ['HIGH','MODERATE','WATCH'], default='OK')
    panel.append(o)

    MODELS[mr] = dict(short=bS, long=bL, phi=phi, cap=cap, tier=tiers[mr],
                      t0=t0, tspan=tspan, hist=op.copy())

    base_mae = mean_absolute_error(y[sp:], [y[:sp].mean()]*(len(y)-sp))
    summary.append({'mailroom':mr,'tier':tiers[mr],'packages':int(vols[mr]),'lockers':cap,
        'open_year_round':mr in OPEN_ALL_YEAR,'modeled_open_days':len(d),
        'dropped_closed_days':n_closed,
        'short_model':bS['label'],'short_mae':round(bS['mae'],2),'short_rmse':round(bS['rmse'],2),
        'short_r2':round(bS['r2'],3),'long_model':bL['label'],'long_mae':round(bL['mae'],2),
        'baseline_mae':round(base_mae,2),
        'mae_improvement_pct':round((1-bS['mae']/base_mae)*100,1),
        'dispersion_phi':round(phi,1),'peak_actual':int(y.max())})

panel_df   = pd.concat(panel, ignore_index=True)
sdf        = pd.DataFrame(summary).sort_values('packages', ascending=False)
lb         = pd.DataFrame(leaderboard)

def _predict_one(c, frame, target, t0, tspan):
    if c['kind']=='hier':
        a = predict(ANCHOR, target)['predicted'] or 0.0
        dw = c['dow_share'].get(pd.Timestamp(target).dayofweek, c['global_share'])
        return max(a*dw, 0.0)
    f = add_features(frame, t0, tspan)
    f = f[f['date']==pd.Timestamp(target)]
    if not len(f): return None
    if c['kind']=='ml':
        return float(max(c['model'].predict(c['scaler'].transform(f[c['feats']].fillna(0).values))[0], 0))
    per = ('move_in' if f['move_in'].iloc[0]==1 else
           'spring_break' if f['spring_break'].iloc[0]==1 else 'session')
    return float(c['table'].get((per, pd.Timestamp(target).dayofweek), c['grand_mean']))

def predict(mailroom, date):
    if mailroom not in MODELS:
        raise ValueError(f"No model for '{mailroom}'. Available: {sorted(MODELS)}")
    M, target = MODELS[mailroom], pd.Timestamp(date).normalize()
    hist, last = M['hist'], M['hist']['date'].max()
    horizon = (target-last).days
    known = None
    if target <= last:
        r = hist[hist['date']==target]
        known = int(r['count'].iloc[0]) if len(r) else None
    base = {'mailroom':mailroom,'date':str(target.date()),'tier':M['tier'],
            'horizon_days':horizon,'lockers':M['cap'],'actual_if_known':known}

    if bool(is_closed(mailroom, pd.Series([target])).iloc[0]):
        return {**base,'is_open':False,'closed_reason':closure_reason(mailroom,target),
                'model_used':'closure_rule','mode':'rule','predicted':0.0,'sd':0.0,
                'lower_95':0.0,'upper_95':0.0,'utilization_pct':0.0,'over_capacity':False}

    c    = M['short'] if horizon <= SHORT_HORIZON else M['long']
    mode = 'short' if horizon <= SHORT_HORIZON else 'long'
    frame = hist.copy()
    if horizon > 0:
        fut = pd.date_range(last+pd.Timedelta(days=1), target)
        fut = fut[~is_closed(mailroom, pd.Series(fut)).values]
        frame = pd.concat([frame, pd.DataFrame({'date':fut,'count':np.nan})], ignore_index=True)
        if mode=='short' and c['kind']=='ml':      # short recursive fill only
            for i in range(len(hist), len(frame)):
                v = _predict_one(c, frame.iloc[:i+1], frame['date'].iloc[i], M['t0'], M['tspan'])
                frame.loc[i,'count'] = 0.0 if v is None else v

    mu = _predict_one(c, frame, target, M['t0'], M['tspan'])
    if mu is None:
        return {**base,'is_open':True,'closed_reason':None,'model_used':c['label'],
                'mode':'unavailable','predicted':None,'sd':None,'lower_95':None,
                'upper_95':None,'utilization_pct':None,'over_capacity':None}
    sd = float(np.sqrt(M['phi']*max(mu,1)))
    return {**base,'is_open':True,'closed_reason':None,'model_used':c['label'],'mode':mode,
            'predicted':round(mu,1),'sd':round(sd,1),
            'lower_95':round(max(mu-1.96*sd,0),1),'upper_95':round(mu+1.96*sd,1),
            'utilization_pct':round(mu/M['cap']*100,1),'over_capacity':bool(mu>M['cap'])}

def predict_range(mailroom, start, end):
    return pd.DataFrame([predict(mailroom,d) for d in pd.date_range(start,end)])

def predict_event(event, year, pad=EVENT_PAD, mailrooms=None):
    anchor = {'christmas':pd.Timestamp(year,12,25),
              'thanksgiving':nth_weekday(year,11,3,4),
              'new_year':pd.Timestamp(year,1,1)}[event]
    rng = pd.date_range(anchor-pd.Timedelta(days=pad), anchor+pd.Timedelta(days=pad))
    return pd.DataFrame([predict(m,d) for m in (mailrooms or sorted(MODELS)) for d in rng])

if __name__ == '__main__':
    pd.set_option('display.width', 340)
    print(sdf[['mailroom','tier','packages','lockers','short_model','short_mae',
               'baseline_mae','mae_improvement_pct','short_rmse','short_r2']].to_string(index=False))
