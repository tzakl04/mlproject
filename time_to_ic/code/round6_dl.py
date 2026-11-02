"""Round 6: operator-free deep models. usage: python round6_dl.py <workdir> <name> [fold_file]"""
import sys, os, json; sys.path.insert(0, '/home/claude/work/src'); os.environ['W'] = sys.argv[1]
import numpy as np, dl2
W = sys.argv[1]; name = sys.argv[2]; ff = sys.argv[3] if len(sys.argv) > 3 else 'folds.json'
cfgs = {'cnn_pre': dict(arch='cnn', seg='arch', pretrain='pre_cnn.pt'), 'pn_pre': dict(arch='pointnet', seg='arch', pretrain='pre_pn.pt'),
        'cnn_neck_pre': dict(arch='cnn', seg='neck', pretrain='pre_cnn.pt'), 'cnn_pre_joint': dict(arch='cnn', seg='arch', pretrain='pre_cnn.pt', target='joint')}
cfg = {**cfgs[name], 'reps': 5, 'threads': 1, 'fold_file': ff}
out, preds = dl2.run(cfg); print(json.dumps({k: v for k, v in out.items() if k != 'cfg'}))
tag = '' if ff == 'folds.json' else '_confirm'; json.dump(out, open(f'{W}/round6/{name}{tag}.json', 'w')); np.save(f'{W}/round6/{name}{tag}.npy', preds)
