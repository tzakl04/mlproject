"""Round 4: sequence models on direction-change sequences (see prep_seq.py).
Reuses dl2's training loop (folds, residual target, augmentation copies, TTA) with new
inputs and three encoders: 1D CNN, bidirectional GRU, Transformer.
cfg keys added: chans in {turn, tan, all}, arch in {cnn, gru, trans}, pool (stem avg-pool).
usage: python dl_seq.py '<json cfg>' name
"""
import sys, json, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, torch, torch.nn as nn
import dl2
from dl2 import CNN1D, W
CH = {'turn': [3, 4, 5], 'tan': [0, 1, 2, 5], 'all': [0, 1, 2, 3, 4, 5], 'turn2': [3, 4]}
_cache = {}
def ZS(key):
    if key not in _cache:
        part = 'corpus' if key.startswith('u_') else key.split('_', 1)[1]
        _cache[key] = np.load(f'{W}/dl_seq_{part}.npz')[key]
    return _cache[key]

class GRUEnc(nn.Module):
    def __init__(s, cin, width, depth, drop, k=None, pool=2):
        super().__init__(); s.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        s.g = nn.GRU(cin, width, num_layers=max(depth, 1), batch_first=True, bidirectional=True, dropout=drop if depth > 1 else 0)
        s.d = nn.Dropout(drop); s.out = 4 * width + 1
    def forward(s, x, l):
        h, _ = s.g(s.pool(x).transpose(1, 2)); h = s.d(h)
        return torch.cat([h.mean(1), h.amax(1), l[:, None]], 1)

class TransEnc(nn.Module):
    def __init__(s, cin, width, depth, drop, k=None, pool=2, n=256):
        super().__init__(); s.pool = nn.AvgPool1d(pool) if pool > 1 else nn.Identity()
        s.emb = nn.Linear(cin, width); s.pos = nn.Parameter(torch.randn(1, n // pool, width) * 0.02)
        layer = nn.TransformerEncoderLayer(width, 4, 2 * width, drop, batch_first=True, norm_first=True)
        s.t = nn.TransformerEncoder(layer, max(depth, 1)); s.d = nn.Dropout(drop); s.out = 2 * width + 1
    def forward(s, x, l):
        h = s.emb(s.pool(x).transpose(1, 2)) + s.pos; h = s.d(s.t(h))
        return torch.cat([h.mean(1), h.amax(1), l[:, None]], 1)

def make_enc(cfg, cin=None):
    cin = len(CH[cfg['chans']]); E = {'cnn': CNN1D, 'gru': GRUEnc, 'trans': TransEnc}[cfg['arch']]
    kw = {} if cfg['arch'] == 'cnn' else {'pool': cfg.get('pool', 2)}
    return E(cin, cfg['width'], cfg['depth'], cfg['drop'], cfg['kernel'], **kw)

def get_inputs(cfg):
    seg = cfg['seg']; ch = CH[cfg['chans']]
    return np.ascontiguousarray(ZS(f'seq_{seg}')[:, :, ch]), ZS(f'logL_{seg}'), None

# patch dl2 so run_fold treats every arch like the 1D-profile branch
dl2.make_enc = make_enc; dl2.get_inputs = get_inputs
# run_fold tests cfg['arch'] in ('cnn','dual') to pick the profile path; alias our archs to 'cnn' at that point
def run(cfg):
    cfg = {**dl2.DEF, 'chans': 'turn', 'seg': 'arch', **cfg}; real = cfg['arch']
    cfg['_real_arch'] = real
    def enc(c, cin=None): return make_enc({**c, 'arch': real}, cin)
    dl2.make_enc = enc; cfg['arch'] = 'cnn'
    out, preds = dl2.run(cfg); out['cfg']['arch'] = real; return out, preds

if __name__ == '__main__':
    cfg = json.loads(sys.argv[1]); name = sys.argv[2] if len(sys.argv) > 2 else 'tmp'
    out, preds = run(cfg); print(json.dumps({k: v for k, v in out.items() if k != 'cfg'}))
    os.makedirs(f'{W}/sweep', exist_ok=True)
    json.dump(out, open(f'{W}/sweep/{name}.json', 'w')); np.save(f'{W}/sweep/{name}.npy', preds)
