"""Round 4 inputs: each path as a SEQUENCE OF DIRECTION CHANGES.

For every (clean or augmented) path, resampled to N equal arc-length steps:
  t_i   unit tangent of step i, in a canonical frame (PCA axes of the path, signs fixed
        by start->end direction)                                   channels 0-2
  k1,k2 turn vector between consecutive tangents, expressed in the parallel-transport
        (Bishop) frame of the previous step: rotation-invariant "how much and which way
        the vessel bends here"                                     channels 3-4
  r     lumen radius                                               channel 5
plus logL (log path length / 300 mm) as a scalar. Augmented copies reuse prep_dl2.augment.
Output: dl_seq.npz with seq_<seg> (n, K+1, 6, N), logL_<seg>, and an unlabeled corpus u_seq.
"""
import sys, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd
from prep_dl2 import load, augment, clean, resample, K_LAB, K_UNL, D, W
from geom import arclen
N = 256

def canon(xyz):
    c = xyz - xyz.mean(0); u, s, vt = np.linalg.svd(c, full_matrices=False); R = vt
    d = xyz[-1] - xyz[0]
    if (R[0] @ d) < 0: R[0] = -R[0]
    if (R[1] @ (xyz[0] - xyz.mean(0))) < 0: R[1] = -R[1]
    R[2] = np.cross(R[0], R[1])
    return c @ R.T

def turn_vectors(t):
    """t: (N,3) unit tangents -> (N,2) turn components in a parallel-transport frame."""
    n = len(t); k = np.zeros((n, 2))
    # initial normal: any unit vector orthogonal to t0
    a = np.array([1.0, 0, 0]) if abs(t[0, 0]) < 0.9 else np.array([0, 1.0, 0])
    nrm = a - (a @ t[0]) * t[0]; nrm /= np.linalg.norm(nrm); b = np.cross(t[0], nrm)
    for i in range(1, n):
        rot = np.cross(t[i - 1], t[i])                      # rotation vector, |rot| = sin(turn angle)
        k[i] = [rot @ nrm, rot @ b]
        # transport the frame: rotate (nrm, b) by the same rotation that takes t[i-1] to t[i]
        c = t[i - 1] @ t[i]; s = np.linalg.norm(rot)
        if s > 1e-9:
            ax = rot / s
            for v in (nrm, b):
                v[:] = v * c + np.cross(ax, v) * s + ax * (ax @ v) * (1 - c)
        nrm -= (nrm @ t[i]) * t[i]; nrm /= np.linalg.norm(nrm); b = np.cross(t[i], nrm)
    return k

def encode_seq(q, lo, hi):
    r = resample(q, N + 1, lo, hi); xyz = canon(r[:, :3])
    t = np.diff(xyz, axis=0); t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    k = turn_vectors(t) / (arclen(r[:, :3])[-1] / N)        # per mm: curvature components
    k = np.clip(k, -1, 1) * 10
    feat = np.concatenate([t, k, r[1:, 3:4] / 3.0], 1).T.astype(np.float32)
    return feat, np.float32(np.log(arclen(r[:, :3])[-1] / 300))

def encode(p, K, strength=1.0):
    S = np.zeros((K + 1, 6, N), np.float32); L = np.zeros(K + 1, np.float32)
    for a in range(K + 1):
        q, lo, hi = clean(p) if a == 0 else augment(p, strength)
        S[a], L[a] = encode_seq(q, lo, hi)
    return S, L

if __name__ == '__main__':
    f = pd.read_csv(f'{W}/features.csv'); c = pd.read_csv(f'{W}/cohort102.csv').set_index('STAR_ID'); out = {}
    part = sys.argv[1] if len(sys.argv) > 1 else 'all'   # arch|archneck|neck|corpus|all -> dl_seq_<part>.npz
    for seg in [s for s in ['arch', 'archneck', 'neck'] if part in (s, 'all')]:
        S = np.zeros((len(f), K_LAB + 1, 6, N), np.float32); L = np.zeros((len(f), K_LAB + 1), np.float32)
        for i, sid in enumerate(f.STAR_ID): S[i], L[i] = encode(load(sid, seg, int(c.at[sid, 'route'])), K_LAB)
        out[f'seq_{seg}'], out[f'logL_{seg}'] = S, L; print(seg, 'labeled done', flush=True)
    U, UL = [], []
    for d in sorted(os.listdir(D)) if part in ('corpus', 'all') else []:
        if not os.path.isdir(f'{D}/{d}'): continue
        for seg in ['archneck', 'arch', 'neck', 'total']:
            for r in range(4):
                try: p = load(d, seg, r)
                except Exception: p = None
                if p is None: continue
                s, l = encode(p, K_UNL); U.append(s); UL.append(l)
    if U: out['u_seq'], out['u_logL'] = np.stack(U), np.stack(UL)
    np.savez_compressed(f'{W}/dl_seq_{part}.npz', **out); print('corpus', len(U), flush=True)
