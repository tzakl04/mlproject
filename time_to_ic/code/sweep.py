"""Round-2 config sweep. Runs each config on the baseline folds (3 repeats for speed),
writes sweep/<name>.json. usage: python sweep.py <stage>"""
import sys, json, subprocess, os, itertools
W = '/home/claude/work'; stage = sys.argv[1]
def run(name, cfg):
    if os.path.exists(f'{W}/sweep/{name}.json'): return
    print('>>', name, cfg, flush=True)
    import time
    while cfg.get('pretrain') and not os.path.exists(f"{W}/{cfg['pretrain']}"): time.sleep(30)
    subprocess.run(['python3', f'{W}/src/dl2.py', json.dumps(cfg), name], stdout=open(f'{W}/sweep/{name}.log', 'w'), stderr=subprocess.STDOUT)
os.makedirs(f'{W}/sweep', exist_ok=True)
base = dict(reps=3, threads=2, seg='arch')  # arch segment won the first block
cb = {'pretrain': 'pre_cnn.pt'}; pb = {'pretrain': 'pre_pn.pt'}  # pretraining helped both on arch
if stage == 'cnn':
    # target form, segment, pretraining
    for tgt in ['residual', 'joint']:
        for seg in ['archneck', 'arch']:
            for pre in [None, 'pre_cnn.pt']:
                run(f'cnn_{tgt}_{seg}_{"pre" if pre else "scratch"}', {**base, 'arch': 'cnn', 'target': tgt, 'seg': seg, 'pretrain': pre})
    # augmentation / regularisation knobs on the residual archneck model
    for af in [0.0, 0.5]: run(f'cnn_augfrac{af}', {**base, **cb, 'arch': 'cnn', 'aug_frac': af})
    for mx in [0.4]: run(f'cnn_mixup{mx}', {**base, **cb, 'arch': 'cnn', 'mixup': mx})
    for w, d in [(16, 3), (64, 4), (32, 5)]: run(f'cnn_w{w}d{d}', {**base, **cb, 'arch': 'cnn', 'width': w, 'depth': d})
    for k in [15]: run(f'cnn_k{k}', {**base, **cb, 'arch': 'cnn', 'kernel': k})
    for lr, wd in [(5e-4, 1e-2), (2e-3, 1e-1), (5e-3, 1e-2)]: run(f'cnn_lr{lr}_wd{wd}', {**base, **cb, 'arch': 'cnn', 'lr': lr, 'wd': wd})
    for dr in [0.5]: run(f'cnn_drop{dr}', {**base, **cb, 'arch': 'cnn', 'drop': dr})
    for ep in [40, 300]: run(f'cnn_ep{ep}', {**base, **cb, 'arch': 'cnn', 'epochs': ep})
    run('cnn_seeds3', {**base, **cb, 'arch': 'cnn', 'n_seeds': 3})
elif stage == 'pn':
    for tgt in ['residual', 'joint']:
        for seg in ['archneck', 'arch']:
            for pre in [None, 'pre_pn.pt']:
                run(f'pn_{tgt}_{seg}_{"pre" if pre else "scratch"}', {**base, 'arch': 'pointnet', 'target': tgt, 'seg': seg, 'pretrain': pre})
    for rot, jit, sc, dp in [(0, 0, 0, 0), (45, 0.005, 0.15, 0.2), (180, 0.002, 0.08, 0.1)]:
        run(f'pn_rot{rot}_jit{jit}_sc{sc}_dp{dp}', {**base, **pb, 'arch': 'pointnet', 'pn_rot': rot, 'pn_jit': jit, 'pn_scale': sc, 'pn_drop': dp})
    for mx in [0.4]: run(f'pn_mixup{mx}', {**base, **pb, 'arch': 'pointnet', 'mixup': mx})
    for w, d in [(16, 3), (64, 4)]: run(f'pn_w{w}d{d}', {**base, **pb, 'arch': 'pointnet', 'width': w, 'depth': d})
    for lr, wd in [(5e-4, 1e-2), (2e-3, 1e-1)]: run(f'pn_lr{lr}_wd{wd}', {**base, **pb, 'arch': 'pointnet', 'lr': lr, 'wd': wd})
    for ep in [40, 300]: run(f'pn_ep{ep}', {**base, **pb, 'arch': 'pointnet', 'epochs': ep})
    run('pn_seeds3', {**base, **pb, 'arch': 'pointnet', 'n_seeds': 3})
