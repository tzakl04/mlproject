"""Round 3, part A: diagnostics and new framings on the feature table.
Everything here is fast (ridge-class models); outputs go to round3/."""
import sys, json, os, warnings; sys.path.insert(0, '/home/claude/work/src'); warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import RidgeCV, ElasticNetCV, LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from cv import score
W = '/home/claude/work'; O = f'{W}/round3'; os.makedirs(O, exist_ok=True)
f = pd.read_csv(f'{W}/features.csv'); y = f.y.values; ymin = f.y_min.values; n = len(f)
F = json.load(open(f'{W}/folds.json')); Fc = json.load(open(f'{W}/folds_confirm.json'))
clin = ['radial', 'left', 'age', 'male', 'op_Ryan', 'op_Min', 'op_Avery', 'op_Unknown']
hc_arch = [c for c in f if c.startswith('hc_arch_') and c != 'hc_arch_TIG2D'] + ['hc_arch_TIG2D']
hc_all = [c for c in f if c.startswith('hc_')]; tab = [c for c in f if c.startswith('tab_')]
X = f.copy()
for c in [c for c in f if 'TIG2D' in c]: X[c] = np.log(X[c].clip(lower=1))
out = {}

def oof(model_fn, cols, folds, target=y):
    P = []
    for fold in folds:
        fold = np.array(fold); p = np.zeros(n)
        for j in range(5):
            tr, te = fold != j, fold == j
            m = model_fn(); m.fit(X.loc[tr, cols].values, target[tr]); p[te] = m.predict(X.loc[te, cols].values)
        P.append(p)
    return np.array(P)
from sklearn.base import clone
def pipe(est): return lambda: make_pipeline(SimpleImputer(), StandardScaler(), clone(est))
ridge = pipe(RidgeCV(alphas=np.logspace(-2, 3, 20)))
def summ(P, target=y, tmin=ymin): d = pd.DataFrame([score(target, p, tmin) for p in P]); return {k: float(d[k].mean()) for k in d}

# 1. variance decomposition (in-sample R2 of log time)
def r2(cols):
    Z = np.column_stack([np.ones(n), X[cols].fillna(X[cols].median()).values]); b = np.linalg.lstsq(Z, y, rcond=None)[0]; return float(1 - ((y - Z @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum())
def fdr(p):
    p = np.asarray(p); o = np.argsort(p); q = np.empty(len(p)); q[o] = np.minimum.accumulate((p[o] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]; return np.clip(q, 0, 1)
out['r2'] = {'operator': r2(['op_Ryan', 'op_Min', 'op_Avery']), 'approach+side': r2(['radial', 'left']), 'age+sex': r2(['age', 'male']),
             'clinical (all)': r2(clin), 'clinical+difficulty ratings': r2(clin + ['arch_diff', 'neck_diff']), 'clinical+arch hand-crafted': r2(clin + hc_arch), 'difficulty ratings only': r2(['arch_diff', 'neck_diff'])}
print('R2', {k: round(v, 3) for k, v in out['r2'].items()})

# 2. ceiling chain: residual after clinical; difficulty ratings as hindsight expert; can morphology predict the ratings?
P_clin = oof(ridge, clin, F); res = y - P_clin.mean(0)
out['resid_rho'] = {'arch_diff': float(spearmanr(res, f.arch_diff, nan_policy='omit')[0]), 'neck_diff': float(spearmanr(res, f.neck_diff)[0]),
                    'arch+neck diff (sum)': float(spearmanr(res, f.arch_diff.fillna(2) + f.neck_diff)[0])}
for tgt, cols in [('arch_diff', hc_arch), ('neck_diff', [c for c in hc_all if '_neck_' in c])]:
    t = f[tgt].fillna(f[tgt].median()).values.astype(float)
    P = oof(ridge, cols, F, target=t); out[f'predict_{tgt}_from_morph'] = float(np.mean([spearmanr(t, p)[0] for p in P]))
    P2 = oof(ridge, clin + cols, F, target=t); out[f'predict_{tgt}_from_clin+morph'] = float(np.mean([spearmanr(t, p)[0] for p in P2]))
    P3 = oof(ridge, clin, F, target=t); out[f'predict_{tgt}_from_clin'] = float(np.mean([spearmanr(t, p)[0] for p in P3]))
json.dump(out, open(f'{O}/diagnostics.json', 'w'), indent=1); print('ceiling', {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items() if 'diff' in k or 'resid' in k})

# 3. univariate screen against the residual, FDR
rows = []
for c in hc_all + tab:
    v = X[c].values; m = np.isfinite(v)
    r, p = spearmanr(res[m], v[m]); rows.append(dict(feature=c, rho_resid=r, p=p, rho_raw=spearmanr(y[m], v[m])[0]))
u = pd.DataFrame(rows); u['q'] = fdr(u.p); u = u.sort_values('p'); u.to_csv(f'{O}/univariate_residual.csv', index=False)
print(u.head(8).round(3).to_string(index=False))

# 4. richer feature models on clinical + arch hand-crafted (and + all hand-crafted), baseline folds
models = {
    'ridge': ridge,
    'elastic net': pipe(ElasticNetCV(l1_ratio=[.2, .5, .8, .95], cv=5, max_iter=5000)),
    'kernel ridge rbf': lambda: make_pipeline(SimpleImputer(), StandardScaler(), GridSearchCV(KernelRidge(kernel='rbf'), {'alpha': [0.1, 1, 10], 'gamma': [0.01, 0.03, 0.1]}, cv=5)),
    'pls (3 comp)': pipe(PLSRegression(3)),
    'hist gradient boosting': lambda: make_pipeline(SimpleImputer(), HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=8)),
    'random forest': lambda: make_pipeline(SimpleImputer(), RandomForestRegressor(300, min_samples_leaf=5, random_state=0)),
}
rows = []
for fs_name, cols in [('clinical', clin), ('clinical+arch hc', clin + hc_arch), ('clinical+all hc', clin + hc_all), ('clinical+arch hc+lab table', clin + hc_arch + tab)]:
    for mn, mf in models.items():
        if mn == 'pls (3 comp)' and len(cols) < 4: continue
        P = oof(mf, cols, F); s = summ(P); rows.append(dict(features=fs_name, model=mn, **s)); print(fs_name, mn, {k: round(v, 3) for k, v in s.items()}, flush=True)
        np.save(f'{O}/oof_{fs_name}_{mn}.npy'.replace(' ', '_').replace('+', '_'), P)
# two-stage: clinical ridge, then morphology model on the residual (per fold)
def two_stage(morph_cols, est_fn):
    P = []
    for fold in F:
        fold = np.array(fold); p = np.zeros(n)
        for j in range(5):
            tr, te = fold != j, fold == j
            m1 = ridge(); m1.fit(X.loc[tr, clin].values, y[tr]); r_tr = y[tr] - m1.predict(X.loc[tr, clin].values)
            m2 = est_fn(); m2.fit(X.loc[tr, morph_cols].values, r_tr); p[te] = m1.predict(X.loc[te, clin].values) + m2.predict(X.loc[te, morph_cols].values)
        P.append(p)
    return np.array(P)
for mn in ['ridge', 'kernel ridge rbf', 'hist gradient boosting']:
    P = two_stage(hc_arch, models[mn]); s = summ(P); rows.append(dict(features='two-stage: clinical ridge + arch hc residual', model=mn, **s)); print('two-stage', mn, {k: round(v, 3) for k, v in s.items()}, flush=True)
pd.DataFrame(rows).to_csv(f'{O}/feature_models.csv', index=False)

# 5. classification framing: slow access (> 10 min, and top quartile > 12 min) from the regression OOF scores on fresh folds
base_c = oof(ridge, clin, Fc); arch_c = oof(ridge, clin + hc_arch, Fc)
ens = np.load(f'{W}/confirm/ensemble.npy'); cnn = np.load(f'{W}/confirm/cnn_residual_arch_pre.npy')
cls = {}
for thr in [10, 12]:
    lab = (ymin > thr).astype(int)
    cls[f'>{thr} min (n={lab.sum()})'] = {k: float(np.mean([roc_auc_score(lab, p) for p in P])) for k, P in [('clinical ridge', base_c), ('clinical+arch hc ridge', arch_c), ('pretrained CNN', cnn), ('CNN+PointNet ensemble', ens)]}
out['auc_fresh_folds'] = cls; print('AUC', cls); json.dump(out, open(f'{O}/diagnostics.json', 'w'), indent=1)

# 6. within-operator: residual correlation per operator, fresh folds
ops = f[['op_Ryan', 'op_Min', 'op_Avery', 'op_Unknown']].idxmax(1).str.replace('op_', '')
wo = {}
for o in ['Ryan', 'Min', 'Avery']:
    m = (ops == o).values; rb = y[m] - base_c.mean(0)[m]
    wo[o] = dict(n=int(m.sum()), rho_arch_hc=float(spearmanr(rb, (arch_c.mean(0) - base_c.mean(0))[m])[0]), rho_cnn=float(spearmanr(rb, (cnn.mean(0) - base_c.mean(0))[m])[0]), rho_ens=float(spearmanr(rb, (ens.mean(0) - base_c.mean(0))[m])[0]),
                 rho_raw_ens=float(spearmanr(y[m], ens.mean(0)[m])[0]), rho_raw_clin=float(spearmanr(y[m], base_c.mean(0)[m])[0]))
out['within_operator'] = wo; print('within-operator', wo)
# leave-one-operator-out: can arch features transfer to an unseen operator (after centring by operator mean of the others)? skip: operator dummies are the model

# 7. learning curve for clinical+arch hc ridge vs clinical, subsampling cases (20 draws)
rng = np.random.default_rng(0); lc = []
for m_ in [40, 60, 80, 101]:
    d = []
    for draw in range(20 if m_ < 101 else 1):
        idx = np.sort(rng.choice(n, m_, replace=False)) if m_ < 101 else np.arange(n)
        sub = f.iloc[idx].reset_index(drop=True); Xs = X.iloc[idx].reset_index(drop=True); ys = y[idx]
        from sklearn.model_selection import StratifiedGroupKFold
        yb = pd.qcut(ys, 4, labels=False, duplicates='drop'); sg = StratifiedGroupKFold(5, shuffle=True, random_state=draw)
        pc = np.zeros(m_); pa = np.zeros(m_)
        for tr, te in sg.split(Xs, yb, sub.group):
            mc = ridge(); mc.fit(Xs.loc[tr, clin].values, ys[tr]); pc[te] = mc.predict(Xs.loc[te, clin].values)
            ma = ridge(); ma.fit(Xs.loc[tr, clin + hc_arch].values, ys[tr]); pa[te] = ma.predict(Xs.loc[te, clin + hc_arch].values)
        d.append((spearmanr(ys, pc)[0], spearmanr(ys, pa)[0]))
    d = np.array(d); lc.append(dict(n=m_, rho_clin=float(d[:, 0].mean()), rho_arch=float(d[:, 1].mean()), delta=float((d[:, 1] - d[:, 0]).mean()), delta_sd=float((d[:, 1] - d[:, 0]).std())))
out['learning_curve'] = lc; print('learning curve', lc)
json.dump(out, open(f'{O}/diagnostics.json', 'w'), indent=1)
