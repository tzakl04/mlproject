import sys, json, subprocess, os, time
W = '/home/claude/work'
def run(name, cfg):
    if os.path.exists(f'{W}/sweep/{name}.json'): return
    while cfg.get('pretrain') and not os.path.exists(f"{W}/{cfg['pretrain']}"): time.sleep(30)
    print('>>', name, cfg, flush=True)
    subprocess.run(['python3', f'{W}/src/dl_seq2.py', json.dumps(cfg), name], stdout=open(f'{W}/sweep/{name}.log', 'w'), stderr=subprocess.STDOUT)
base = dict(reps=3, threads=1, seg='arch', arch='cnn', chans='turn', pretrain='pre_seq_cnn_turn.pt')
for aw in [0.3, 1.0]: run(f'seq2_cnn_aux{aw}', {**base, 'aux_w': aw})
for rw in [0.5, 2.0]: run(f'seq2_cnn_rank{rw}', {**base, 'rank_w': rw})
run('seq2_cnn_aux0.3_rank0.5', {**base, 'aux_w': 0.3, 'rank_w': 0.5})
run('seq2_cnn_archneck_aux0.3', {**base, 'seg': 'archneck', 'aux_w': 0.3})
run('seq2_cnn_aux0.3_seeds3', {**base, 'aux_w': 0.3, 'n_seeds': 3})
