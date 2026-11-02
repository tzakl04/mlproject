"""Round 5 deep models on the extended cohort: (1) pooled CV on 191 with the round-2 recipe, (2) old->new transfer."""
import sys, os, json; sys.path.insert(0, '/home/claude/work/src'); os.environ['W'] = '/home/claude/work_ext'
import numpy as np, pandas as pd, torch
import dl2; from dl2 import run_fold, get_inputs, y, f, C
from cv import score
from sklearn.linear_model import RidgeCV
W = os.environ['W']; O = f'{W}/round5'; mode = sys.argv[1]
if mode == 'transfer':
    torch.set_num_threads(2); old = (f.batch == 'old').values; new = ~old; rows = []
    for name, cfg in [('cnn_residual_arch_pre', dict(arch='cnn', seg='arch', pretrain='pre_cnn.pt')), ('pn_residual_arch_pre', dict(arch='pointnet', seg='arch', pretrain='pre_pn.pt')), ('cnn_residual_arch_scratch', dict(arch='cnn', seg='arch', pretrain=None))]:
        cfg = {**dl2.DEF, **cfg}; prof, logL, cloud = get_inputs(cfg)
        ps = [run_fold(cfg, old, new, 1000 + s, prof, logL, cloud) for s in range(5)]; p = np.mean(ps, 0)
        cm, cs = C[old].mean(0), C[old].std(0) + 1e-6; base = RidgeCV(alphas=np.logspace(-2, 3, 20)).fit((C[old] - cm) / cs, y[old]).predict((C[new] - cm) / cs)
        s = score(y[new], p, f.y_min.values[new]); sb = score(y[new], base, f.y_min.values[new])
        rows.append(dict(model=name, **{k: float(v) for k, v in s.items()}, morph_only_rho=float(pd.Series(p - base).corr(pd.Series(y[new]), method='spearman')), clin_rho=float(sb['spearman'])))
        print(rows[-1], flush=True); np.save(f'{O}/transfer_{name}.npy', np.array(ps))
    pd.DataFrame(rows).to_csv(f'{O}/transfer_dl.csv', index=False)
else:  # pooled CV, name = sys.argv[2]
    name = sys.argv[2]; cfgs = {'cnn_residual_arch_pre': dict(arch='cnn', seg='arch', pretrain='pre_cnn.pt'), 'pn_residual_arch_pre': dict(arch='pointnet', seg='arch', pretrain='pre_pn.pt'), 'cnn_residual_arch_scratch': dict(arch='cnn', seg='arch', pretrain=None)}
    cfg = {**cfgs[name], 'reps': 5, 'threads': 2, 'fold_file': sys.argv[3] if len(sys.argv) > 3 else 'folds.json'}
    out, preds = dl2.run(cfg); print(json.dumps({k: v for k, v in out.items() if k != 'cfg'}))
    tag = '' if cfg['fold_file'] == 'folds.json' else '_confirm'
    json.dump(out, open(f'{O}/cv_{name}{tag}.json', 'w')); np.save(f'{O}/cv_{name}{tag}.npy', preds)
