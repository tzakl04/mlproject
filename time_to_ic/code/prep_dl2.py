"""Build inputs for the second round of deep models.

Labeled (cohort, treated route): clean resampled path per segment plus K augmented
profile copies. Unlabeled (every patient folder, all 4 routes, arch + archneck):
profile copies for self-supervised pretraining. Nothing here uses the outcome.
"""
import sys, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd, scipy.io as sio
from geom import circumR, torsion, arclen
D = '/home/claude/data'; W = '/home/claude/work'
rng = np.random.default_rng(1)
N1, N2 = 256, 512
K_LAB, K_UNL = 60, 16

def smooth(q, w=15):
    ker = np.ones(w) / w; q = q.copy()
    for j in range(3): q[:, j] = np.convolve(np.pad(q[:, j], w // 2, mode='edge'), ker, 'valid')
    return q

def resample(p, n, a=0.0, b=1.0):
    s = arclen(p[:, :3]); L = s[-1]; t = np.linspace(a * L, b * L, n)
    return np.stack([np.interp(t, s, p[:, j]) for j in range(4)], 1)

def rotmat(deg):
    ax = rng.normal(size=3); ax /= np.linalg.norm(ax); th = np.deg2rad(rng.uniform(-deg, deg))
    K_ = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    return np.eye(3) + np.sin(th) * K_ + (1 - np.cos(th)) * K_ @ K_

def profile(q):
    xyz = q[:, :3]; k = 1 / circumR(xyz); k = np.clip(np.nan_to_num(k, nan=0, posinf=0), 0, 1.0)
    tor = np.clip(np.nan_to_num(torsion(xyz)), -1, 1)
    return np.stack([k * 10, tor * 5, q[:, 3] / 3.0], 0).astype(np.float32)

def augment(p, strength=1.0):
    """Geometric augmentation on the raw 0.1 mm path. strength scales every knob."""
    q = p.copy(); s = strength
    sc = rng.uniform(1 - 0.08 * s, 1 + 0.08 * s); R = rotmat(15 * s)
    c0 = q[0, :3]; q[:, :3] = (q[:, :3] - c0) @ R.T * sc + c0
    q[:, 3] *= rng.uniform(1 - 0.05 * s, 1 + 0.05 * s)
    # low-frequency warp: smooth random displacement field along the path (mm)
    n = len(q); m = max(3, int(rng.integers(3, 7)))
    knots = rng.normal(0, 1.5 * s, (m, 3)); t = np.linspace(0, 1, n)
    warp = np.stack([np.interp(t, np.linspace(0, 1, m), knots[:, j]) for j in range(3)], 1)
    q[:, :3] += warp
    q[:, :3] += rng.normal(0, 0.15 * s, (n, 3))
    q = smooth(q, int(rng.integers(11, 25)) | 1)
    a = rng.uniform(0, 0.04 * s); b = 1 - rng.uniform(0, 0.04 * s)
    return q, a, b

def clean(p): return smooth(p), 0.0, 1.0

def load(i, seg, r):
    m = sio.loadmat(f'{D}/{i}/smoothpathssubmmtrunc{seg}ant.mat', squeeze_me=True, struct_as_record=False)
    p = np.asarray(m['smoothPaths'][r].path, float)
    return p if p.ndim == 2 and len(p) > 200 else None

def encode(p, K, strength):
    prof = np.zeros((K + 1, 3, N1), np.float32); logL = np.zeros(K + 1, np.float32); cloud = None
    for a in range(K + 1):
        q, lo, hi = clean(p) if a == 0 else augment(p, strength)
        r1 = resample(q, N1, lo, hi); prof[a] = profile(r1); logL[a] = np.log(arclen(r1[:, :3])[-1] / 300)
        if a == 0:
            r2 = resample(q, N2); cloud = np.concatenate([(r2[:, :3] - r2[0, :3]) / 100.0, r2[:, 3:4] / 3.0], 1).astype(np.float32)
    return prof, logL, cloud

if __name__ == '__main__':
    f = pd.read_csv(f'{W}/features.csv'); c = pd.read_csv(f'{W}/cohort102.csv').set_index('STAR_ID')
    out = {}
    for seg in ['archneck', 'arch']:
        P = np.zeros((len(f), K_LAB + 1, 3, N1), np.float32); L = np.zeros((len(f), K_LAB + 1), np.float32); Cl = np.zeros((len(f), N2, 4), np.float32)
        for i, sid in enumerate(f.STAR_ID):
            p = load(sid, seg, int(c.at[sid, 'route']))
            P[i], L[i], Cl[i] = encode(p, K_LAB, 1.0)
        out[f'prof_{seg}'], out[f'logL_{seg}'], out[f'cloud_{seg}'] = P, L, Cl
        print(seg, 'labeled done', flush=True)
    # unlabeled corpus: every folder, every route, both segments
    U, UL, UC, meta = [], [], [], []
    for d in sorted(os.listdir(D)):
        if not os.path.isdir(f'{D}/{d}'): continue
        for seg in ['archneck', 'arch']:
            for r in range(4):
                try: p = load(d, seg, r)
                except Exception: p = None
                if p is None: continue
                pr, ll, cl = encode(p, K_UNL, 1.0); U.append(pr); UL.append(ll); UC.append(cl); meta.append((int(d), seg, r))
    out['u_prof'], out['u_logL'], out['u_cloud'] = np.stack(U), np.stack(UL), np.stack(UC)
    out['u_meta'] = np.array([(m[0], 0 if m[1] == 'archneck' else 1, m[2]) for m in meta])
    np.savez_compressed(f'{W}/dl_inputs2.npz', **out)
    print('unlabeled paths:', len(U), 'profile shape', out['prof_archneck'].shape)
