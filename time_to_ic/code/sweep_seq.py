"""Round-4 sweep: sequence encoders x channel sets, arch segment, sweep folds (3 reps).
usage: python sweep_seq.py <stage>  (a|b|c)"""
import sys, json, subprocess, os, time
W = '/home/claude/work'; stage = sys.argv[1]
def run(name, cfg):
    if os.path.exists(f'{W}/sweep/{name}.json'): return
    while cfg.get('pretrain') and not os.path.exists(f"{W}/{cfg['pretrain']}"): time.sleep(30)
    print('>>', name, cfg, flush=True)
    subprocess.run(['python3', f'{W}/src/dl_seq.py', json.dumps(cfg), name], stdout=open(f'{W}/sweep/{name}.log', 'w'), stderr=subprocess.STDOUT)
base = dict(reps=3, threads=1, seg='arch', target='residual')
if stage == 'a':   # scratch: encoder x channels
    for ch in ['turn', 'all', 'tan']:
        run(f'seq_cnn_{ch}', {**base, 'arch': 'cnn', 'chans': ch})
        run(f'seq_gru_{ch}', {**base, 'arch': 'gru', 'chans': ch, 'width': 32, 'depth': 2, 'lr': 1e-3})
elif stage == 'b':
    for ch in ['turn', 'all', 'tan']:
        run(f'seq_trans_{ch}', {**base, 'arch': 'trans', 'chans': ch, 'width': 64, 'depth': 2, 'lr': 5e-4, 'drop': 0.2})
    run('seq_gru_turn_w64', {**base, 'arch': 'gru', 'chans': 'turn', 'width': 64, 'depth': 2, 'lr': 1e-3})
    run('seq_cnn_turn_k15', {**base, 'arch': 'cnn', 'chans': 'turn', 'kernel': 15})
elif stage == 'c':  # pretrained encoders
    for a, ch in [('cnn', 'turn'), ('cnn', 'all'), ('gru', 'turn'), ('trans', 'turn')]:
        extra = {'width': 32, 'depth': 2, 'lr': 1e-3} if a == 'gru' else {'width': 64, 'depth': 2, 'lr': 5e-4, 'drop': 0.2} if a == 'trans' else {}
        run(f'seq_{a}_{ch}_pre', {**base, 'arch': a, 'chans': ch, 'pretrain': f'pre_seq_{a}_{ch}.pt', **extra})
