"""Round 5 (extended cohort, 101 old + 90 new): feature-model baselines.
Clinical sets: full = approach, side, age, sex, operator dummies (Unknown for the new batch);
noop = approach, side, age, sex. Subsets: all, old only, new only. Plus old->new transfer."""
import sys, json, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline; from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV; from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import spearmanr
from cv import score
W = '/home/claude/work_ext'; O = f'{W}/round5'
f = pd.read_csv(f'{W}/features.csv')
for k in [c for c in f if 'TIG2D' in c]: f[k] = np.log(f[k].clip(lower=1))
y = f.y.values; ymin = f.y_min.values
clin_full = ['radial', 'left', 'age', 'male', 'op_Ryan', 'op_Min', 'op_Avery', 'op_Unknown']; clin_noop = ['radial', 'left', 'age', 'male']
hc_arch = [c for c in f if c.startswith('hc_arch_')]; hc_core = [f'hc_{s}_{k}' for s in ['arch', 'neck', 'archneck'] for k in ['PL', 'TIG', 'k1_mean', 'k1_p90', 'rad_mean', 'rad_min', 'tor_mean']]
tab = [c for c in f if c.startswith('tab_')]
ridge = lambda: make_pipeline(SimpleImputer(strategy='median'), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30)))
rf = lambda: make_pipeline(SimpleImputer(strategy='median'), RandomForestRegressor(300, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=2))
def folds_for(idx, seed0=100):
    sub = f.iloc[idx]; yb = pd.qcut(sub.y, 5, labels=False, duplicates='drop').values; out = []
    for r in range(5):
        sg = StratifiedGroupKFold(5, shuffle=True, random_state=seed0 + r); fold = np.zeros(len(sub), int)
        for j, (_, te) in enumerate(sg.split(sub, yb, sub.group)): fold[te] = j
        out.append(fold)
    return out
def cv(idx, cols, mk):
    F = folds_for(idx); res = []; preds = []
    for fold in F:
        p = np.zeros(len(idx))
        for j in range(5):
            tr, te = fold != j, fold == j
            if not cols: p[te] = y[idx][tr].mean(); continue
            m = mk(); m.fit(f.iloc[idx[tr]][cols], y[idx][tr]); p[te] = m.predict(f.iloc[idx[te]][cols])
        res.append(score(y[idx], p, ymin[idx])); preds.append(p)
    d = pd.DataFrame(res); return dict(spearman=d.spearman.mean(), spearman_sd=d.spearman.std(), mae_min=d.mae_min.mean()), np.mean(preds, 0)
subsets = {'all (n=191)': np.arange(len(f)), 'old (n=101)': np.where(f.batch == 'old')[0], 'new (n=90)': np.where(f.batch == 'new')[0]}
rows = []; oof = {}
for sn, idx in subsets.items():
    sets = {'null': [], 'clinical full': clin_full, 'clinical no-operator': clin_noop,
            'clinical full + arch hc': clin_full + hc_arch, 'clinical no-op + arch hc': clin_noop + hc_arch,
            'clinical full + all hc': clin_full + hc_core, 'clinical no-op + all hc': clin_noop + hc_core,
            'clinical full + lab table': clin_full + tab, 'arch hc only': hc_arch, 'all hc only': hc_core, 'lab table only': tab,
            'clinical full + difficulty (hindsight)': clin_full + ['arch_diff', 'neck_diff']}
    if sn.startswith('new'): sets = {k: v for k, v in sets.items() if 'full' not in k or k == 'null'}
    for name, cols in sets.items():
        for mn, mk in [('ridge', ridge), ('rf', rf)]:
            if name == 'null' and mn == 'rf': continue
            s, p = cv(idx, cols, mk); rows.append(dict(subset=sn, features=name, model=mn, n_feat=len(cols), **s)); oof[f'{sn}|{name}|{mn}'] = (idx, p)
            print(sn, name, mn, {k: round(v, 3) for k, v in s.items()}, flush=True)
r = pd.DataFrame(rows); r.to_csv(f'{O}/baselines_ext.csv', index=False)
# transfer: fit on old, test on new (no operator available for new)
old, new = subsets['old (n=101)'], subsets['new (n=90)']; trows = []
for name, cols in [('clinical no-operator', clin_noop), ('clinical no-op + arch hc', clin_noop + hc_arch), ('clinical no-op + all hc', clin_noop + hc_core), ('arch hc only', hc_arch), ('all hc only', hc_core)]:
    for mn, mk in [('ridge', ridge), ('rf', rf)]:
        m = mk(); m.fit(f.iloc[old][cols], y[old]); p = m.predict(f.iloc[new][cols]); s = score(y[new], p, ymin[new])
        trows.append(dict(train='old 101', test='new 90', features=name, model=mn, **{k: float(v) for k, v in s.items()})); print('transfer', name, mn, {k: round(float(v), 3) for k, v in s.items()}, flush=True)
pd.DataFrame(trows).to_csv(f'{O}/transfer_old_to_new.csv', index=False)
# univariate screen on the new batch (residual after no-op clinical) and on all
out = {}
for sn, idx, cl in [('new', new, clin_noop), ('all', subsets['all (n=191)'], clin_full)]:
    _, pc = cv(idx, cl, ridge); res = y[idx] - pc; u = []
    for c in hc_core + tab:
        v = f.iloc[idx][c].values; m = np.isfinite(v); rr, pp = spearmanr(res[m], v[m]); u.append(dict(feature=c, rho_resid=rr, p=pp, rho_raw=spearmanr(y[idx][m], v[m])[0]))
    u = pd.DataFrame(u).sort_values('p'); u.to_csv(f'{O}/univariate_{sn}.csv', index=False); print(sn, u.head(6).round(3).to_string(index=False))
