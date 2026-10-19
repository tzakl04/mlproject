"""Collect the round-2 sweep, then re-run the chosen configs on fresh folds (seeds
200-204) against a clinical ridge fitted on the same folds, with a paired bootstrap.

usage: python confirm.py collect            -> sweep_round2.csv
       python confirm.py run name1 name2 .. -> confirm/<name>.json + confirm_round2.csv
"""
import sys, json, glob, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd
from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from cv import score
W = '/home/claude/work'
f = pd.read_csv(f'{W}/features.csv'); y = f.y.values; ymin = f.y_min.values
clin = ['radial', 'left', 'age', 'male', 'op_Ryan', 'op_Min', 'op_Avery', 'op_Unknown']
C = f[clin].astype(float).values; C[np.isnan(C)] = np.nanmedian(C, 0)[np.where(np.isnan(C))[1]]

def ridge_oof(folds):
    out = []
    for fold in folds:
        fold = np.array(fold); p = np.zeros(len(y))
        for j in range(5):
            tr, te = fold != j, fold == j; m, s = C[tr].mean(0), C[tr].std(0) + 1e-6
            p[te] = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((C[tr] - m) / s, y[tr]).predict((C[te] - m) / s)
        out.append(p)
    return np.array(out)

def boot(pa, pb, n=2000, seed=0):
    """paired bootstrap over cases of (rho_a - rho_b) and (mae_a - mae_b), averaged over repeats."""
    rng = np.random.default_rng(seed); R = len(pa); dr, dm = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        dr.append(np.mean([spearmanr(y[i], pa[r][i])[0] - spearmanr(y[i], pb[r][i])[0] for r in range(R)]))
        dm.append(np.mean([np.abs(ymin[i] - np.expm1(pa[r][i])).mean() - np.abs(ymin[i] - np.expm1(pb[r][i])).mean() for r in range(R)]))
    return dict(d_rho=float(np.mean(dr)), d_rho_lo=float(np.percentile(dr, 2.5)), d_rho_hi=float(np.percentile(dr, 97.5)),
                d_mae=float(np.mean(dm)), d_mae_lo=float(np.percentile(dm, 2.5)), d_mae_hi=float(np.percentile(dm, 97.5)))

if sys.argv[1] == 'collect':
    rows = []
    for fn in sorted(glob.glob(f'{W}/sweep/*.json')):
        d = json.load(open(fn)); name = os.path.basename(fn)[:-5]
        if name.startswith('smoke'): continue
        c = d['cfg']; rows.append(dict(name=name, arch=c['arch'], seg=c['seg'], target=c['target'], pretrain=bool(c['pretrain']), width=c['width'], depth=c['depth'],
                                       drop=c['drop'], lr=c['lr'], wd=c['wd'], epochs=c['epochs'], mixup=c['mixup'], aug_frac=c['aug_frac'], n_seeds=c['n_seeds'],
                                       pn_rot=c['pn_rot'], spearman=d['spearman'], spearman_sd=d['spearman_sd'], mae_min=d['mae_min'], mae_min_sd=d['mae_min_sd'], secs=d['secs']))
    t = pd.DataFrame(rows).sort_values('spearman', ascending=False); t.to_csv(f'{W}/sweep_round2.csv', index=False)
    print(t[['name', 'spearman', 'spearman_sd', 'mae_min', 'mae_min_sd', 'secs']].round(3).to_string(index=False))
else:
    import dl2
    F = json.load(open(f'{W}/folds_confirm.json')); base = ridge_oof(F)
    os.makedirs(f'{W}/confirm', exist_ok=True)
    bs = [score(y, p, ymin) for p in base]; print('clinical ridge (confirm folds)', pd.DataFrame(bs).mean().round(3).to_dict())
    rows = [dict(name='clinical ridge', **pd.DataFrame(bs).mean().to_dict())]
    for name in sys.argv[2:]:
        cfg = json.load(open(f'{W}/sweep/{name}.json'))['cfg']; cfg.update(fold_file='folds_confirm.json', reps=5, threads=2)
        out, preds = dl2.run(cfg); np.save(f'{W}/confirm/{name}.npy', preds)
        b = boot(preds, base); out.update(b); json.dump(out, open(f'{W}/confirm/{name}.json', 'w'))
        rows.append(dict(name=name, spearman=out['spearman'], mae_log=out['mae_log'], mae_min=out['mae_min'], **b)); print(name, {k: round(v, 3) for k, v in rows[-1].items() if k != 'name'})
    pd.DataFrame(rows).to_csv(f'{W}/confirm_round2_{sys.argv[2]}.csv', index=False)
