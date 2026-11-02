"""Round 6 feature models, operator-free, on the 191-case cohort.
A. expert difficulty ratings (arch, neck) from morphology + approach/side/age/sex: Spearman, AUC for 'challenging or worse' (>=3).
B. slow access (time to IC > 10 min, and > 12) from the same inputs, and from the predicted-difficulty score (nested OOF)."""
import sys, json; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd
from sklearn.pipeline import make_pipeline; from sklearn.impute import SimpleImputer; from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV, LogisticRegressionCV; from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
W = '/home/claude/work_ext'; O = f'{W}/round6'; import os; os.makedirs(O, exist_ok=True)
f = pd.read_csv(f'{W}/features.csv')
for k in [c for c in f if 'TIG2D' in c]: f[k] = np.log(f[k].clip(lower=1))
base = ['radial', 'left', 'age', 'male']; seg = {s: [c for c in f if c.startswith(f'hc_{s}_')] for s in ['arch', 'neck', 'archneck']}; tab = {s: [c for c in f if c.startswith('tab_') and c.endswith(s.capitalize().replace('neck', 'Neck'))] for s in ['arch', 'neck', 'archneck']}
ridge = lambda: make_pipeline(SimpleImputer(), StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
rf = lambda: make_pipeline(SimpleImputer(), RandomForestRegressor(400, min_samples_leaf=5, max_features=0.5, random_state=0, n_jobs=2))
def folds(sub, strat, seed0=100, n=5):
    out = []
    for r in range(n):
        sg = StratifiedGroupKFold(5, shuffle=True, random_state=seed0 + r); fold = np.zeros(len(sub), int)
        for j, (_, te) in enumerate(sg.split(sub, strat, sub.group)): fold[te] = j
        out.append(fold)
    return out
def oof(sub, cols, t, mk, strat):
    P = []
    for fold in folds(sub, strat):
        p = np.zeros(len(sub))
        for j in range(5):
            tr, te = fold != j, fold == j; m = mk(); m.fit(sub.loc[tr, cols], t[tr]); p[te] = m.predict(sub.loc[te, cols])
        P.append(p)
    return np.array(P)
def boot_auc(lab, P, n=1000):
    rng = np.random.default_rng(0); v = []
    for _ in range(n):
        i = rng.integers(0, len(lab), len(lab))
        if lab[i].min() == lab[i].max(): continue
        v.append(np.mean([roc_auc_score(lab[i], p[i]) for p in P]))
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
rows = []; scores = {}
# A. difficulty ratings
for tgt, s in [('arch_diff', 'arch'), ('neck_diff', 'neck'), ('neck_diff', 'archneck')]:
    sub = f[f[tgt].notna()].reset_index(drop=True); t = sub[tgt].values.astype(float); lab = (t >= 3).astype(int); strat = t.astype(int)
    for name, cols in [('approach/side/age/sex', base), (f'{s} hand-crafted', seg[s]), (f'{s} hand-crafted + base', seg[s] + base), (f'{s} lab table + base', tab[s] + base), ('all hand-crafted + base', seg['arch'] + seg['neck'] + base)]:
        for mn, mk in [('ridge', ridge), ('rf', rf)]:
            P = oof(sub, cols, t, mk, strat); rho = float(np.mean([spearmanr(t, p)[0] for p in P])); auc = float(np.mean([roc_auc_score(lab, p) for p in P])); lo, hi = boot_auc(lab, P)
            rows.append(dict(target=f'{tgt} (n={len(sub)}, >=3: {lab.sum()})', features=name, model=mn, spearman=rho, auc_difficult=auc, auc_lo=lo, auc_hi=hi)); print(rows[-1], flush=True)
            if mn == 'ridge' and name == f'{s} hand-crafted + base': scores[f'{tgt}_{s}'] = (sub.STAR_ID.values, P.mean(0))
pd.DataFrame(rows).to_csv(f'{O}/difficulty_models.csv', index=False)
# B. slow access, operator-free, all 191
rows = []; sub = f.reset_index(drop=True); t = sub.y.values; strat = pd.qcut(sub.y, 5, labels=False, duplicates='drop').values
pred_diff = {}  # nested: predicted difficulty score for every case from a model fitted on the training fold only (ratings used only in training)
def oof_diffscore(cols, tgt):
    P = []
    for fold in folds(sub, strat):
        p = np.zeros(len(sub))
        for j in range(5):
            tr, te = fold != j, fold == j; trr = tr & sub[tgt].notna().values; m = ridge(); m.fit(sub.loc[trr, cols], sub.loc[trr, tgt].values.astype(float)); p[te] = m.predict(sub.loc[te, cols])
        P.append(p)
    return np.array(P)
PA = oof_diffscore(seg['arch'] + base, 'arch_diff'); PN = oof_diffscore(seg['neck'] + base, 'neck_diff')
sub['pred_arch_diff'] = PA.mean(0); sub['pred_neck_diff'] = PN.mean(0)
for name, cols in [('approach/side/age/sex', base), ('arch hand-crafted + base', seg['arch'] + base), ('arch+neck hand-crafted + base', seg['arch'] + seg['neck'] + base), ('predicted arch+neck difficulty + base', ['pred_arch_diff', 'pred_neck_diff'] + base), ('hindsight ratings + base (reference)', ['arch_diff', 'neck_diff'] + base)]:
    for mn, mk in [('ridge', ridge), ('rf', rf)]:
        P = oof(sub, cols, t, mk, strat); rho = float(np.mean([spearmanr(t, p)[0] for p in P])); r = dict(features=name, model=mn, spearman_time=rho)
        for thr in [10, 12]:
            lab = (sub.y_min.values > thr).astype(int); r[f'auc_>{thr}'] = float(np.mean([roc_auc_score(lab, p) for p in P])); r[f'auc_>{thr}_lo'], r[f'auc_>{thr}_hi'] = boot_auc(lab, P)
        rows.append(r); print(r, flush=True)
pd.DataFrame(rows).to_csv(f'{O}/slow_access_models.csv', index=False)
print('pred arch diff vs time rho', round(spearmanr(sub.y, sub.pred_arch_diff)[0], 3), 'pred neck diff vs time', round(spearmanr(sub.y, sub.pred_neck_diff)[0], 3), 'hindsight arch vs time', round(spearmanr(sub.y, sub.arch_diff, nan_policy='omit')[0], 3))
sub[['STAR_ID', 'batch', 'pred_arch_diff', 'pred_neck_diff']].to_csv(f'{O}/pred_difficulty_oof.csv', index=False)
