"""Contrastive (SimCLR-style) pretraining of the path encoder on every route of every
patient folder (no outcome labels used). Two augmented views of the same path are
positives; everything else in the batch is a negative.

usage: python pretrain.py cnn|pointnet out.pt [epochs]
"""
import sys, json; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from dl2 import CNN1D, PointNet, pn_aug, DEF, W, Z
torch.set_num_threads(4); torch.manual_seed(0); rng = np.random.default_rng(0); g = torch.Generator().manual_seed(0)
arch = sys.argv[1]; out = sys.argv[2]; E = int(sys.argv[3]) if len(sys.argv) > 3 else 60
cfg = {**DEF, 'arch': arch}
cfg.update(json.loads(sys.argv[4]) if len(sys.argv) > 4 else {})
U, UL, UC = Z['u_prof'], Z['u_logL'], torch.tensor(Z['u_cloud']); n, K = U.shape[:2]
enc = (CNN1D if arch == 'cnn' else PointNet)(3 if arch == 'cnn' else 4, cfg['width'], cfg['depth'], 0.0, cfg['kernel'])
proj = nn.Sequential(nn.Linear(enc.out, 64), nn.ReLU(), nn.Linear(64, 32))
opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=2e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E); bs = 64; tau = 0.1
def view(bi):
    if arch == 'cnn':
        ai = rng.integers(1, K, len(bi)); return torch.tensor(U[bi, ai]), torch.tensor(UL[bi, ai])
    return pn_aug(UC[bi], cfg, g), torch.tensor(UL[bi, 0])
for ep in range(E):
    enc.train(); perm = rng.permutation(n); tot = 0; nb = 0
    for b0 in range(0, n - bs + 1, bs):
        bi = perm[b0:b0 + bs]
        z = torch.cat([proj(enc(*view(bi))), proj(enc(*view(bi)))]); z = Fn.normalize(z, dim=1)
        sim = z @ z.T / tau; sim.fill_diagonal_(-1e9)
        tgt = torch.cat([torch.arange(bs) + bs, torch.arange(bs)])
        loss = Fn.cross_entropy(sim, tgt); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
    sched.step()
    if ep % 10 == 0 or ep == E - 1: print(arch, 'epoch', ep, 'loss', round(tot / nb, 3), flush=True)
torch.save(enc.state_dict(), f'{W}/{out}')
