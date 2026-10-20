"""Round-2 deep models: configurable 1D CNN / PointNet with heavier augmentation,
optional contrastive pretraining on the unlabeled route corpus, residual-on-clinical
target, and seed ensembling. Same folds as the baselines.

usage: python dl2.py '<json config>' [out_name]
"""
import sys, json, time, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from sklearn.linear_model import RidgeCV
from cv import score
W = '/home/claude/work'
DEF = dict(arch='cnn', seg='archneck', width=32, depth=4, drop=0.3, lr=2e-3, wd=1e-2, epochs=120, bs=16,
           target='residual', aug_frac=1.0, mixup=0.0, pretrain=None, n_seeds=1, reps=5, fold_file='folds.json',
           tta=8, pn_rot=15, pn_jit=0.002, pn_scale=0.08, pn_drop=0.1, clin=True, threads=4, kernel=7)

f = pd.read_csv(f'{W}/features.csv')
clin_cols = ['radial', 'left', 'age', 'male', 'op_Ryan', 'op_Min', 'op_Avery', 'op_Unknown']
C = f[clin_cols].astype(float).values; C[np.isnan(C)] = np.nanmedian(C, 0)[np.where(np.isnan(C))[1]]
y = f.y.values.astype(np.float32)
Z = np.load(f'{W}/dl_inputs3.npz' if os.path.exists(f'{W}/dl_inputs3.npz') else f'{W}/dl_inputs2.npz')

class CNN1D(nn.Module):
    def __init__(s, cin, width, depth, drop, k):
        super().__init__(); layers = []; i = cin
        for d in range(depth):
            o = width if d else width // 2
            layers += [nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU(), nn.MaxPool1d(2)]; i = o
        s.f = nn.Sequential(*layers, nn.Dropout(drop)); s.out = 2 * i + 1
    def forward(s, x, l): h = s.f(x); return torch.cat([h.mean(2), h.amax(2), l[:, None]], 1)

class DualCNN(nn.Module):
    """Two 1D CNNs (arch, neck) on stacked channels; features concatenated."""
    def __init__(s, cin, width, depth, drop, k):
        super().__init__(); s.a = CNN1D(3, width, depth, drop, k); s.b = CNN1D(3, width, depth, drop, k); s.out = s.a.out + s.b.out
    def forward(s, x, l): return torch.cat([s.a(x[:, :3], l), s.b(x[:, 3:], l)], 1)

class PointNet(nn.Module):
    def __init__(s, cin, width, depth, drop, k=None):
        super().__init__(); layers = []; i = cin
        for d in range(depth):
            o = width * (2 if d == depth - 1 else 1)
            layers += [nn.Conv1d(i, o, 1), nn.BatchNorm1d(o), nn.ReLU()]; i = o
        s.f = nn.Sequential(*layers, nn.Dropout(drop)); s.out = 2 * i + 1
    def forward(s, x, l): h = s.f(x); return torch.cat([h.amax(2), h.mean(2), l[:, None]], 1)

class Net(nn.Module):
    def __init__(s, enc, nclin, drop, opgain=False):
        super().__init__(); s.enc = enc
        s.head = nn.Sequential(nn.Linear(enc.out, 32), nn.ReLU(), nn.Dropout(drop), nn.Linear(32, 1))
        s.lin = nn.Linear(nclin, 1) if nclin else None
        s.gain = nn.Linear(4, 1) if opgain else None  # per-operator scaling of the morphology term
        if opgain: nn.init.zeros_(s.gain.weight); nn.init.ones_(s.gain.bias)
    def forward(s, x, l, c):
        o = s.head(s.enc(x, l))[:, 0]
        if s.gain is not None: o = o * s.gain(c[:, -4:])[:, 0]
        if s.lin is not None: o = o + s.lin(c)[:, 0]
        return o

def make_enc(cfg, cin):
    E = CNN1D if cfg['arch'] == 'cnn' else DualCNN if cfg['arch'] == 'dual' else PointNet
    return E(cin, cfg['width'], cfg['depth'], cfg['drop'], cfg['kernel'])

def rot_batch(n, deg, g):
    ax = torch.randn(n, 3, generator=g); ax = ax / ax.norm(dim=1, keepdim=True)
    th = (torch.rand(n, generator=g) * 2 - 1) * np.deg2rad(deg)
    K = torch.zeros(n, 3, 3); K[:, 0, 1] = -ax[:, 2]; K[:, 0, 2] = ax[:, 1]; K[:, 1, 0] = ax[:, 2]; K[:, 1, 2] = -ax[:, 0]; K[:, 2, 0] = -ax[:, 1]; K[:, 2, 1] = ax[:, 0]
    return torch.eye(3)[None] + torch.sin(th)[:, None, None] * K + (1 - torch.cos(th))[:, None, None] * K @ K

def pn_aug(x, cfg, g):
    """x: (n, 512, 4) clean clouds -> (n, 4, 512) augmented."""
    n = x.shape[0]; xyz = x[:, :, :3]; r = x[:, :, 3:]
    R = rot_batch(n, cfg['pn_rot'], g); sc = 1 + (torch.rand(n, 1, 1, generator=g) * 2 - 1) * cfg['pn_scale']
    xyz = torch.bmm(xyz, R.transpose(1, 2)) * sc + torch.randn(xyz.shape, generator=g) * cfg['pn_jit']
    r = r * (1 + (torch.rand(n, 1, 1, generator=g) * 2 - 1) * 0.05)
    out = torch.cat([xyz, r], 2)
    if cfg['pn_drop'] > 0:  # random point dropout (replace with first point like PointNet++)
        m = torch.rand(n, out.shape[1], 1, generator=g) < cfg['pn_drop']
        out = torch.where(m, out[:, :1, :], out)
    return out.permute(0, 2, 1).contiguous()

def get_inputs(cfg):
    seg = cfg['seg']
    if cfg['arch'] == 'dual': return np.concatenate([Z['prof_arch'], Z['prof_neck']], 2), Z['logL_arch'], None
    if cfg['arch'] == 'cnn': return Z[f'prof_{seg}'], Z[f'logL_{seg}'], None
    return None, Z[f'logL_{seg}'], torch.tensor(Z[f'cloud_{seg}'])

def run_fold(cfg, tr, te, seed, prof, logL, cloud):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); g = torch.Generator().manual_seed(seed)
    tri, tei = np.where(tr)[0], np.where(te)[0]
    cm, cs = C[tr].mean(0), C[tr].std(0) + 1e-6; Cn = torch.tensor((C - cm) / cs, dtype=torch.float32)
    base = np.zeros(len(y), np.float32)
    if cfg['target'] == 'residual':
        rr = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((C[tr] - cm) / cs, y[tr]); base = rr.predict((C - cm) / cs).astype(np.float32)
    t = y - base; mu, sd = t[tr].mean(), t[tr].std() + 1e-6; yt = torch.tensor((t - mu) / sd)
    nclin = C.shape[1] if (cfg['clin'] and cfg['target'] != 'residual') else 0
    net = Net(make_enc(cfg, 6 if cfg['arch'] == 'dual' else 3 if cfg['arch'] == 'cnn' else 4), nclin, cfg['drop'], cfg.get('opgain', False))
    if cfg['pretrain']:
        sdict = torch.load(f"{W}/{cfg['pretrain']}")
        if cfg['arch'] == 'dual': net.enc.a.load_state_dict(sdict, strict=False); net.enc.b.load_state_dict(sdict, strict=False)
        else: net.enc.load_state_dict(sdict, strict=False)
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    E = cfg['epochs']; sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg['lr'], total_steps=E * ((len(tri) + cfg['bs'] - 1) // cfg['bs']), pct_start=0.15)
    lossf = nn.HuberLoss(delta=1.0); K = prof.shape[1] if prof is not None else 0
    def batch(bi, train):
        l = logL[bi, 0]
        if cfg['arch'] in ('cnn', 'dual'):
            ai = rng.integers(1, K, len(bi)) if train and K > 1 else np.zeros(len(bi), int)
            if train: ai = np.where(rng.random(len(bi)) < cfg['aug_frac'], ai, 0)
            x = torch.tensor(prof[bi, ai]); l = logL[bi, ai]
        else:
            x = pn_aug(cloud[bi], cfg, g) if train else cloud[bi].permute(0, 2, 1).contiguous()
        return x, torch.tensor(l), Cn[bi]
    for ep in range(E):
        net.train(); perm = rng.permutation(tri)
        for b0 in range(0, len(perm), cfg['bs']):
            bi = perm[b0:b0 + cfg['bs']]
            if len(bi) < 2: continue
            x, l, c = batch(bi, True); yy = yt[bi]
            if cfg['mixup'] > 0:
                lam = float(rng.beta(cfg['mixup'], cfg['mixup'])); j = torch.randperm(len(bi), generator=g)
                x = lam * x + (1 - lam) * x[j]; l = lam * l + (1 - lam) * l[j]; c = lam * c + (1 - lam) * c[j]; yy = lam * yy + (1 - lam) * yy[j]
            loss = lossf(net(x, l, c), yy); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); ps = []
    with torch.no_grad():
        x, l, c = batch(tei, False); ps.append(net(x, l, c).numpy())
        for a in range(cfg['tta']):
            if cfg['arch'] in ('cnn', 'dual'):
                ai = np.full(len(tei), 1 + a % max(K - 1, 1)); x = torch.tensor(prof[tei, ai]); l = torch.tensor(logL[tei, ai])
            else:
                x = pn_aug(cloud[tei], cfg, g)
            ps.append(net(x, l, c).numpy())
    return np.mean(ps, 0) * sd + mu + base[tei]

def run(cfg):
    cfg = {**DEF, **cfg}; torch.set_num_threads(cfg['threads'])
    prof, logL, cloud = get_inputs(cfg); F = json.load(open(f"{W}/{cfg['fold_file']}"))
    res, preds, t0 = [], [], time.time()
    for rep, fold in enumerate(F[:cfg['reps']]):
        fold = np.array(fold); pred = np.zeros(len(f))
        for j in range(5):
            pred[fold == j] = np.mean([run_fold(cfg, fold != j, fold == j, rep * 10 + j + 1000 * s, prof, logL, cloud) for s in range(cfg['n_seeds'])], 0)
        s = score(y, pred, f.y_min.values); res.append(s); preds.append(pred)
        print('rep', rep, {k: round(float(v), 3) for k, v in s.items()}, f'{time.time() - t0:.0f}s', flush=True)
    d = pd.DataFrame(res)
    return dict(cfg=cfg, spearman=float(d.spearman.mean()), spearman_sd=float(d.spearman.std()), mae_min=float(d.mae_min.mean()),
                mae_min_sd=float(d.mae_min.std()), mae_log=float(d.mae_log.mean()), secs=time.time() - t0), np.array(preds)

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else 'tmp'
    out, preds = run(cfg); print(json.dumps({k: v for k, v in out.items() if k != 'cfg'}))
    os.makedirs(f'{W}/sweep', exist_ok=True)
    json.dump(out, open(f'{W}/sweep/{name}.json', 'w')); np.save(f'{W}/sweep/{name}.npy', preds)
