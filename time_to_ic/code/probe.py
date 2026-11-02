"""Linear probe: frozen pretrained encoder embeddings (arch) + clinical -> ridge, same folds.
Also tries the embedding of both segments. usage: python probe.py"""
import sys, json, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd, torch
from sklearn.linear_model import RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from dl2 import CNN1D, PointNet, DEF, W, Z, C, y, f
from cv import score
torch.set_num_threads(2)
F = json.load(open(f'{W}/folds.json')); Fc = json.load(open(f'{W}/folds_confirm.json')); n = len(y); ymin = f.y_min.values
def embed(arch, seg, ckpt):
    enc = (CNN1D if arch == 'cnn' else PointNet)(3 if arch == 'cnn' else 4, DEF['width'], DEF['depth'], 0.0, DEF['kernel'])
    enc.load_state_dict(torch.load(f'{W}/{ckpt}')); enc.eval()
    with torch.no_grad():
        if arch == 'cnn':
            P, L = Z[f'prof_{seg}'], Z[f'logL_{seg}']  # average embedding over the clean copy + 8 augmented copies
            E = np.mean([enc(torch.tensor(P[:, a]), torch.tensor(L[:, a])).numpy() for a in range(9)], 0)
        else:
            E = enc(torch.tensor(Z[f'cloud_{seg}']).permute(0, 2, 1).contiguous(), torch.tensor(Z[f'logL_{seg}'][:, 0])).numpy()
    return E
def run(Xm, folds, k=None):
    P = []
    for fold in folds:
        fold = np.array(fold); p = np.zeros(n)
        for j in range(5):
            tr, te = fold != j, fold == j
            sc = StandardScaler().fit(Xm[tr]); A, B = sc.transform(Xm[tr]), sc.transform(Xm[te])
            if k:
                pca = PCA(k).fit(A[:, C.shape[1]:]); A = np.hstack([A[:, :C.shape[1]], pca.transform(A[:, C.shape[1]:])]); B = np.hstack([B[:, :C.shape[1]], pca.transform(B[:, C.shape[1]:])])
            m = RidgeCV(alphas=np.logspace(-1, 4, 25)).fit(A, y[tr]); p[te] = m.predict(B)
        P.append(p)
    P = np.array(P); d = pd.DataFrame([score(y, p, ymin) for p in P]); return {k_: float(d[k_].mean()) for k_ in d}, P
rows = []
embs = {('cnn', 'arch'): embed('cnn', 'arch', 'pre_cnn.pt'), ('pointnet', 'arch'): embed('pointnet', 'arch', 'pre_pn.pt'),
        ('cnn', 'archneck'): embed('cnn', 'archneck', 'pre_cnn.pt'), ('pointnet', 'archneck'): embed('pointnet', 'archneck', 'pre_pn.pt')}
for (a, s), E in embs.items():
    for k in [None, 5, 10]:
        for fname, folds in [('sweep', F), ('fresh', Fc)]:
            r, P = run(np.hstack([C, E]), folds, k); rows.append(dict(encoder=a, seg=s, pca=k or 'none', folds=fname, **r)); print(a, s, k, fname, {q: round(v, 3) for q, v in r.items()}, flush=True)
            if fname == 'fresh' and k == 10 and s == 'arch': np.save(f'{W}/round3/oof_probe_{a}_arch.npy', P)
E2 = np.hstack([embs[('cnn', 'arch')], embs[('pointnet', 'arch')]])
for fname, folds in [('sweep', F), ('fresh', Fc)]:
    r, P = run(np.hstack([C, E2]), folds, 10); rows.append(dict(encoder='cnn+pointnet', seg='arch', pca=10, folds=fname, **r)); print('both', fname, r)
r, _ = run(C, Fc); rows.append(dict(encoder='none (clinical)', seg='-', pca='-', folds='fresh', **r)); r, _ = run(C, F); rows.append(dict(encoder='none (clinical)', seg='-', pca='-', folds='sweep', **r))
pd.DataFrame(rows).to_csv(f'{W}/round3/probe.csv', index=False)
