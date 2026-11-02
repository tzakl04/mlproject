"""SimCLR pretraining of a sequence encoder (cnn|gru|trans) on the unlabeled direction-change corpus.
usage: python pretrain_seq.py <arch> <chans> out.pt [epochs] [json cfg]"""
import sys, json; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, torch, torch.nn as nn, torch.nn.functional as Fn
from dl2 import DEF, W
from dl_seq import make_enc, CH, ZS
torch.set_num_threads(2); torch.manual_seed(0); rng = np.random.default_rng(0)
arch, chans, out = sys.argv[1:4]; E = int(sys.argv[4]) if len(sys.argv) > 4 else 60
cfg = {**DEF, 'arch': arch, 'chans': chans}; cfg.update(json.loads(sys.argv[5]) if len(sys.argv) > 5 else {})
U = np.ascontiguousarray(ZS('u_seq')[:, :, CH[chans]]); UL = ZS('u_logL'); n, K = U.shape[:2]
enc = make_enc(cfg); proj = nn.Sequential(nn.Linear(enc.out, 64), nn.ReLU(), nn.Linear(64, 32))
opt = torch.optim.AdamW(list(enc.parameters()) + list(proj.parameters()), lr=1e-3, weight_decay=1e-2)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, E); bs = 64; tau = 0.1
def view(bi): ai = rng.integers(1, K, len(bi)); return torch.tensor(U[bi, ai]), torch.tensor(UL[bi, ai])
for ep in range(E):
    enc.train(); perm = rng.permutation(n); tot = 0; nb = 0
    for b0 in range(0, n - bs + 1, bs):
        bi = perm[b0:b0 + bs]
        z = torch.cat([proj(enc(*view(bi))), proj(enc(*view(bi)))]); z = Fn.normalize(z, dim=1)
        sim = z @ z.T / tau; sim.fill_diagonal_(-1e9); tgt = torch.cat([torch.arange(bs) + bs, torch.arange(bs)])
        loss = Fn.cross_entropy(sim, tgt); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item(); nb += 1
    sched.step()
    if ep % 10 == 0 or ep == E - 1: print(arch, chans, 'epoch', ep, 'loss', round(tot / nb, 3), flush=True)
torch.save(enc.state_dict(), f'{W}/{out}')
