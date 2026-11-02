"""Labeled round-2 inputs (arch profiles + clouds) for the extended cohort; unlabeled corpus copied from dl_inputs3.npz."""
import sys, os; sys.path.insert(0, '/home/claude/work/src'); os.environ.setdefault('W', '/home/claude/work_ext')
import numpy as np, pandas as pd
from prep_dl2 import load, encode, K_LAB, N1, N2
W = os.environ['W']; part = sys.argv[1]  # arch | neck | archneck
f = pd.read_csv(f'{W}/features.csv'); c = pd.read_csv(f'{W}/cohort102.csv').set_index('STAR_ID')
P = np.zeros((len(f), K_LAB + 1, 3, N1), np.float32); L = np.zeros((len(f), K_LAB + 1), np.float32); Cl = np.zeros((len(f), N2, 4), np.float32)
for i, sid in enumerate(f.STAR_ID):
    try: P[i], L[i], Cl[i] = encode(load(sid, part, int(c.at[sid, 'route'])), K_LAB, 1.0)
    except Exception as e: print('skip', sid, part, str(e)[:50]); P[i] = np.nan
np.savez_compressed(f'{W}/dl_lab_{part}.npz', prof=P, logL=L, cloud=Cl); print(part, 'done', np.isnan(P[:, 0, 0, 0]).sum(), 'missing')
