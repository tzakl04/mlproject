"""Extend round-2 inputs: labeled neck profiles (for the dual arch+neck encoder) and a bigger
unlabeled corpus (arch, neck, archneck, total; all 4 routes; every folder) for pretraining."""
import sys, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd
from prep_dl2 import load, encode, K_LAB, K_UNL, N1, N2, D, W
f = pd.read_csv(f'{W}/features.csv'); c = pd.read_csv(f'{W}/cohort102.csv').set_index('STAR_ID')
Z = dict(np.load(f'{W}/dl_inputs2.npz'))
P = np.zeros((len(f), K_LAB + 1, 3, N1), np.float32); L = np.zeros((len(f), K_LAB + 1), np.float32); Cl = np.zeros((len(f), N2, 4), np.float32)
for i, sid in enumerate(f.STAR_ID):
    P[i], L[i], Cl[i] = encode(load(sid, 'neck', int(c.at[sid, 'route'])), K_LAB, 1.0)
Z['prof_neck'], Z['logL_neck'], Z['cloud_neck'] = P, L, Cl; print('neck labeled done', flush=True)
U, UL, UC = [], [], []
for d in sorted(os.listdir(D)):
    if not os.path.isdir(f'{D}/{d}'): continue
    for seg in ['archneck', 'arch', 'neck', 'total']:
        for r in range(4):
            try: p = load(d, seg, r)
            except Exception: p = None
            if p is None: continue
            pr, ll, cl = encode(p, K_UNL, 1.0); U.append(pr); UL.append(ll); UC.append(cl)
Z['u2_prof'], Z['u2_logL'], Z['u2_cloud'] = np.stack(U), np.stack(UL), np.stack(UC)
np.savez_compressed(f'{W}/dl_inputs3.npz', **Z); print('big corpus', len(U))
