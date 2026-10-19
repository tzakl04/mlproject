"""Round-2 figures: sweep overview and fresh-fold confirmation vs clinical ridge."""
import sys, json, glob, os; sys.path.insert(0, '/home/claude/work/src')
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
W = '/home/claude/work'; OUT = '/mnt/project-files/time-to-ic-cnn'
BG, INK, MUTED = '#fcfcfb', '#0b0b0b', '#52514e'; COL = {'cnn': '#2a78d6', 'pointnet': '#eb6834'}
sw = pd.read_csv(f'{W}/sweep_round2.csv')
conf = pd.concat([pd.read_csv(f) for f in glob.glob(f'{W}/confirm_round2_*.csv')]).drop_duplicates('name')
ens = json.load(open(f'{W}/confirm/ensemble.json')); conf = pd.concat([conf, pd.DataFrame([dict(name='ensemble cnn+pointnet', **ens)])])
CLIN_SWEEP = 0.517  # clinical ridge on the sweep folds (5 repeats)
fig, ax = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG, gridspec_kw=dict(width_ratios=[1.3, 1]))
# left: every sweep config
s = sw.sort_values('spearman').reset_index(drop=True); yy = np.arange(len(s))
for a in ['cnn', 'pointnet']:
    m = s.arch == a; ax[0].errorbar(s.spearman[m], yy[m], xerr=s.spearman_sd[m], fmt='o', color=COL[a], ms=6, lw=1.5, label={'cnn': '1D CNN', 'pointnet': 'Point-cloud net'}[a])
ax[0].axvline(CLIN_SWEEP, color='#999', lw=1, ls='--'); ax[0].text(CLIN_SWEEP, len(s) - 0.3, ' clinical ridge', color=MUTED, fontsize=9)
lab = s.name.str.replace('_', ' ').str.replace('residual', 'resid').str.replace(' pre', ' pretrained').str.replace('scratch', 'no pretrain')
ax[0].set_yticks(yy, lab, fontsize=8); ax[0].set_xlabel('Spearman ρ on the sweep folds (3 repeats, ±1 SD)', color=MUTED)
ax[0].set_title(f'Sweep: {len(s)} configurations', loc='left', color=INK, fontsize=11); ax[0].legend(frameon=False, loc='lower right')
# right: confirmation on fresh folds, delta vs clinical ridge
c = conf[conf.name != 'clinical ridge'].sort_values('d_rho').reset_index(drop=True); yy = np.arange(len(c))
col = [COL['cnn'] if 'cnn' in n and 'pointnet' not in n else COL['pointnet'] if 'pn_' in n else '#1baf7a' for n in c.name]
ax[1].errorbar(c.d_rho, yy, xerr=[c.d_rho - c.d_rho_lo, c.d_rho_hi - c.d_rho], fmt='o', ecolor='#999', color='none', ms=0, lw=1.5)
ax[1].scatter(c.d_rho, yy, c=col, s=60, zorder=3)
for i, r in c.iterrows(): ax[1].text(c.d_rho_hi.max() + 0.01, i, f'ρ {r.spearman:.3f}, MAE {r.mae_min:.2f} min', va='center', fontsize=8, color=MUTED)
ax[1].axvline(0, color='#999', lw=1)
ax[1].set_yticks(yy, c.name.str.replace('_', ' ').str.replace('residual', 'resid').str.replace(' pre', ' pretrained').str.replace('scratch', 'no pretrain'), fontsize=8)
ax[1].set_xlabel('Δ Spearman ρ vs clinical ridge, fresh folds (95% bootstrap CI)', color=MUTED); ax[1].set_xlim(c.d_rho_lo.min() - 0.02, c.d_rho_hi.max() + 0.16)
ax[1].set_title('Confirmation on fresh folds (5 repeats)\nclinical ridge: ρ 0.505, MAE 5.28 min', loc='left', color=INK, fontsize=11)
for a in ax:
    a.set_facecolor(BG); a.grid(axis='x', color='#e5e4e0', lw=0.8); a.set_axisbelow(True)
    for k, sp in a.spines.items():
        if k in ('top', 'right'): sp.set_visible(False)
fig.suptitle('Round 2: pretrained encoders on the arch, residual on clinical. Gains are small and their CIs include zero.', x=0.01, ha='left', color=INK, fontsize=11)
plt.tight_layout(); plt.savefig(f'{OUT}/plots/round2_sweep_confirm.png', dpi=150)
conf.to_csv(f'{OUT}/results_round2_confirm.csv', index=False); sw.to_csv(f'{OUT}/results_round2_sweep.csv', index=False)
print(conf.round(3).to_string(index=False))
