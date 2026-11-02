"""Round 4b: sequence models + (a) auxiliary supervision from the operators' hindsight
difficulty ratings (arch_diff, neck_diff) and (b) a pairwise ranking loss (Spearman is the metric).
cfg keys added: aux_w (weight of the rating heads), rank_w (weight of the pairwise logistic rank loss).
usage: python dl_seq2.py '<json cfg>' name"""
import sys, json, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import RidgeCV
import dl2, dl_seq
from dl2 import W, C, y, f
A = f[['arch_diff', 'neck_diff']].values.astype(np.float32)   # NaN allowed

class Net2(nn.Module):
    def __init__(s, enc, drop):
        super().__init__(); s.enc = enc
        s.head = nn.Sequential(nn.Linear(enc.out, 32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32, 1))
        s.aux = nn.Linear(enc.out, 2)
    def forward(s, x, l, c=None):
        h = s.enc(x, l); return s.head(h)[:, 0], s.aux(h)

def run_fold(cfg, tr, te, seed, prof, logL, cloud):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    tri, tei = np.where(tr)[0], np.where(te)[0]
    cm, cs = C[tr].mean(0), C[tr].std(0) + 1e-6
    rr = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((C[tr] - cm) / cs, y[tr]); base = rr.predict((C - cm) / cs).astype(np.float32)
    t = y - base; mu, sd = t[tr].mean(), t[tr].std() + 1e-6; yt = torch.tensor((t - mu) / sd)
    am, asd = np.nanmean(A[tr], 0), np.nanstd(A[tr], 0) + 1e-6; At = torch.tensor(np.nan_to_num((A - am) / asd)); Am = torch.tensor(np.isfinite(A).astype(np.float32))
    net = Net2(dl_seq.make_enc({**cfg, 'arch': cfg['_real_arch']}), cfg['drop'])
    if cfg['pretrain']: net.enc.load_state_dict(torch.load(f"{W}/{cfg['pretrain']}"), strict=False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    E = cfg['epochs']; sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg['lr'], total_steps=E * ((len(tri) + cfg['bs'] - 1) // cfg['bs']), pct_start=0.15)
    lossf = nn.HuberLoss(delta=1.0); K = prof.shape[1]
    def batch(bi, train):
        ai = rng.integers(1, K, len(bi)) if train else np.zeros(len(bi), int)
        return torch.tensor(prof[bi, ai]), torch.tensor(logL[bi, ai])
    for ep in range(E):
        net.train(); perm = rng.permutation(tri)
        for b0 in range(0, len(perm), cfg['bs']):
            bi = perm[b0:b0 + cfg['bs']]
            if len(bi) < 2: continue
            x, l = batch(bi, True); yy = yt[bi]; o, a = net(x, l)
            loss = lossf(o, yy)
            if cfg['aux_w'] > 0: loss = loss + cfg['aux_w'] * (((a - At[bi]) ** 2) * Am[bi]).sum() / Am[bi].sum().clamp(min=1)
            if cfg['rank_w'] > 0:
                d = o[:, None] - o[None, :]; s = torch.sign(yy[:, None] - yy[None, :])
                loss = loss + cfg['rank_w'] * (torch.nn.functional.softplus(-d * s) * (s != 0)).sum() / (s != 0).sum().clamp(min=1)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); ps = []
    with torch.no_grad():
        x, l = batch(tei, False); ps.append(net(x, l)[0].numpy())
        for a_ in range(cfg['tta']):
            ai = np.full(len(tei), 1 + a_ % max(K - 1, 1)); ps.append(net(torch.tensor(prof[tei, ai]), torch.tensor(logL[tei, ai]))[0].numpy())
    return np.mean(ps, 0) * sd + mu + base[tei]

def run(cfg):
    cfg = {**dl2.DEF, 'chans': 'turn', 'seg': 'arch', 'aux_w': 0.0, 'rank_w': 0.0, **cfg}; real = cfg['arch']; cfg['_real_arch'] = real; cfg['arch'] = 'cnn'
    dl2.run_fold = run_fold; dl2.get_inputs = dl_seq.get_inputs
    out, preds = dl2.run(cfg); out['cfg']['arch'] = real; return out, preds

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]); name = sys.argv[2]
    out, preds = run(cfg); print(json.dumps({k: v for k, v in out.items() if k != 'cfg'}))
    json.dump(out, open(f'{W}/sweep/{name}.json', 'w')); np.save(f'{W}/sweep/{name}.npy', preds)
