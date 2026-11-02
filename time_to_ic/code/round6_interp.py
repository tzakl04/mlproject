"""Round 6 interpretability on the 191-case cohort (arch segment, clean paths):
(1) positional analysis: Spearman of local curvature / turn magnitude at each normalised arc position vs arch rating and vs time;
(2) arch-type clustering: k-means on the canonical turn-vector sequence (PCA-reduced), cluster vs rating / time / approach."""
import sys, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr, kruskal
from sklearn.decomposition import PCA; from sklearn.cluster import KMeans
from prep_seq import encode_seq, N
from prep_dl2 import clean
W = '/home/claude/work_ext'; O = f'{W}/round6'
f = pd.read_csv(f'{W}/features.csv'); P = np.load(f'{W}/paths.npy', allow_pickle=True).item()
S = np.zeros((len(f), 6, N), np.float32); L = np.zeros(len(f), np.float32)
for i, sid in enumerate(f.STAR_ID):
    q, lo, hi = clean(P[f'{sid}_arch']); S[i], L[i] = encode_seq(q, lo, hi)
np.savez_compressed(f'{O}/arch_seq_clean.npz', seq=S, logL=L, STAR_ID=f.STAR_ID.values)
turn = np.hypot(S[:, 3], S[:, 4])                       # |turn vector| per step (curvature magnitude, x10 per mm)
# smooth over 16 steps (~1/16 of the arch) for positional stats
ker = np.ones(16) / 16; T = np.stack([np.convolve(t, ker, 'same') for t in turn])
rows = []
for pos in range(0, N, 8):
    r = dict(pos=pos / N)
    for tgt in ['arch_diff', 'y']:
        m = f[tgt].notna().values; rho, p = spearmanr(f[tgt].values[m], T[m, pos]); r[f'rho_{tgt}'] = rho; r[f'p_{tgt}'] = p
    r['mean_turn'] = float(T[:, pos].mean()); rows.append(r)
pos = pd.DataFrame(rows); pos.to_csv(f'{O}/positional_curvature.csv', index=False)
print(pos.round(3).to_string(index=False))
# arch types: cluster on the resampled tangent + turn sequence (rotation handled by canonical frame), 64 positions
X = np.concatenate([S[:, :3, ::4].reshape(len(f), -1), S[:, 3:5, ::4].reshape(len(f), -1), L[:, None]], 1)
X = (X - X.mean(0)) / (X.std(0) + 1e-6); Z = PCA(10, random_state=0).fit_transform(X)
out = {}
for k in [3, 4]:
    km = KMeans(k, n_init=20, random_state=0).fit(Z); lab = km.labels_; f[f'arch_cluster_k{k}'] = lab
    g = f.groupby(lab).agg(n=('y', 'size'), time_median=('y_min', 'median'), arch_diff_mean=('arch_diff', 'mean'), frac_difficult=('arch_diff', lambda v: (v >= 3).mean()), radial=('radial', 'mean'), left=('left', 'mean'), age=('age', 'mean'), PL=('hc_arch_PL', 'mean'), TIG=('hc_arch_TIG', 'mean'))
    kw_t = kruskal(*[f.y[lab == c] for c in range(k)]).pvalue; m = f.arch_diff.notna(); kw_d = kruskal(*[f.arch_diff[m & (lab == c)] for c in range(k)]).pvalue
    out[k] = dict(table=g.round(2).to_dict(), kruskal_p_time=float(kw_t), kruskal_p_archdiff=float(kw_d)); print(k, 'clusters'); print(g.round(2).to_string()); print('kruskal p time', round(kw_t, 4), 'p arch rating', round(kw_d, 4))
    np.save(f'{O}/cluster_centres_k{k}.npy', np.stack([S[lab == c].mean(0) for c in range(k)]))
json.dump(out, open(f'{O}/arch_clusters.json', 'w'), indent=1); f[['STAR_ID', 'batch', 'arch_cluster_k3', 'arch_cluster_k4']].to_csv(f'{O}/arch_clusters.csv', index=False)
