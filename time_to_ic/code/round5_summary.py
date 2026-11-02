import sys, os, json, glob; sys.path.insert(0, '/home/claude/work/src'); os.environ['W'] = '/home/claude/work_ext'
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from cv import score
import confirm as cf   # ridge_oof, boot (uses W env)
W = '/home/claude/work_ext'; O = f'{W}/round5'; f = pd.read_csv(f'{W}/features.csv'); y = f.y.values; ymin = f.y_min.values
old = (f.batch == 'old').values; new = ~old; rows = []
for tag, ff in [('sweep folds', 'folds.json'), ('fresh folds', 'folds_confirm.json')]:
    F = json.load(open(f'{W}/{ff}')); base = cf.ridge_oof(F)
    def summ(P, m): d = pd.DataFrame([score(y[m], p[m], ymin[m]) for p in P]); return {k: float(d[k].mean()) for k in ['spearman', 'mae_min']}
    rows.append(dict(folds=tag, model='clinical ridge (operator unknown for new)', **summ(base, slice(None)), rho_old=summ(base, old)['spearman'], rho_new=summ(base, new)['spearman']))
    for name in ['cnn_residual_arch_pre', 'pn_residual_arch_pre', 'cnn_residual_arch_scratch']:
        fn = f'{O}/cv_{name}{"" if ff == "folds.json" else "_confirm"}.npy'
        if not os.path.exists(fn): continue
        P = np.load(fn); b = cf.boot(P, base, 1000)
        rows.append(dict(folds=tag, model=name, **summ(P, slice(None)), rho_old=summ(P, old)['spearman'], rho_new=summ(P, new)['spearman'], **{k: round(v, 3) for k, v in b.items()},
                         resid_rho_old=float(spearmanr((y - base.mean(0))[old], (P.mean(0) - base.mean(0))[old])[0]), resid_rho_new=float(spearmanr((y - base.mean(0))[new], (P.mean(0) - base.mean(0))[new])[0])))
    if ff == 'folds_confirm.json' and all(os.path.exists(f'{O}/cv_{n}_confirm.npy') for n in ['cnn_residual_arch_pre', 'pn_residual_arch_pre']):
        E = (np.load(f'{O}/cv_cnn_residual_arch_pre_confirm.npy') + np.load(f'{O}/cv_pn_residual_arch_pre_confirm.npy')) / 2; b = cf.boot(E, base, 1000)
        rows.append(dict(folds=tag, model='cnn+pn ensemble', **summ(E, slice(None)), rho_old=summ(E, old)['spearman'], rho_new=summ(E, new)['spearman'], **{k: round(v, 3) for k, v in b.items()}))
t = pd.DataFrame(rows); print(t.round(3).to_string(index=False)); t.to_csv(f'{O}/deep_ext.csv', index=False)
