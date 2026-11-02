import json, numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.stats import spearmanr
OUT='/mnt/project-files/time-to-ic-cnn'
b=pd.read_csv('results_baselines.csv',keep_default_na=False)
b=b[b.features!='null'].copy()
null_mae=pd.read_csv('results_baselines.csv',keep_default_na=False).query("features=='null'").mae_min.astype(float).iloc[0]
b['label']=b.features+' ('+b.model.replace({'ridge':'ridge','rf':'forest'})+')'
d=pd.DataFrame([json.load(open(f'res_{a}_{c}.json')) for a in ['cnn','pointnet'] for c in ['clin','only']])
d['label']=d.features.replace({'cnn +clinical':'1D CNN + clinical','cnn only':'1D CNN only','pointnet +clinical':'Point-cloud net + clinical','pointnet only':'Point-cloud net only'})
r=pd.concat([b,d],ignore_index=True)
for k in ['spearman','spearman_sd','mae_min','mae_min_sd','mae_log']: r[k]=r[k].astype(float)
def kind(l):
    if 'difficulty' in l: return 'Reference (post-hoc ratings)'
    if 'CNN' in l or 'Point' in l: return 'Deep learning'
    if l.startswith('clinical ('): return 'Clinical only'
    return 'Morphology features'
r['kind']=r.label.map(kind)
r.to_csv(f'{OUT}/results_table.csv',index=False)
col={'Clinical only':'#2a78d6','Morphology features':'#eb6834','Deep learning':'#1baf7a','Reference (post-hoc ratings)':'#eda100'}
r=r.sort_values('spearman').reset_index(drop=True)
fig,ax=plt.subplots(1,2,figsize=(12,6.5),sharey=True,facecolor='#fcfcfb')
y=np.arange(len(r))
for k,c in col.items():
    m=(r.kind==k).values
    ax[0].errorbar(r.spearman[m],y[m],xerr=r.spearman_sd[m],fmt='o',color=c,ms=8,lw=2,label=k)
    ax[1].errorbar(r.mae_min[m],y[m],xerr=r.mae_min_sd[m],fmt='o',color=c,ms=8,lw=2)
ax[0].set_yticks(y); ax[0].set_yticklabels(r.label,color='#0b0b0b')
ax[0].axvline(0,color='#999',lw=1); ax[0].set_xlabel('Spearman ρ, predicted vs actual (higher is better)',color='#52514e')
ax[1].axvline(null_mae,color='#999',lw=1,ls='--'); ax[1].text(null_mae,len(r)-0.6,' predict the mean',color='#52514e',fontsize=9)
ax[1].set_xlabel('Mean absolute error, minutes (lower is better)',color='#52514e')
for a in ax:
    a.set_facecolor('#fcfcfb'); a.grid(axis='x',color='#e5e5e2'); a.tick_params(colors='#52514e')
    for k2,s in a.spines.items():
        if k2 in('top','right'): s.set_visible(False)
ax[0].legend(loc='lower right',frameon=False,fontsize=9)
fig.suptitle('Time to IC access: 5-fold CV × 5 repeats, same folds for every model (n = 101). Bars = ±1 SD across repeats.',x=0.01,ha='left',color='#0b0b0b',fontsize=11)
plt.tight_layout(); plt.savefig(f'{OUT}/plots/model_comparison.png',dpi=150)
f=pd.read_csv('features.csv'); o=pd.read_csv('oof_baselines.csv')
P={'Clinical (ridge)':o['clinical|ridge'],'Clinical + arch features (ridge)':o['clinical+arch hand-crafted|ridge'],'Point-cloud net + clinical':np.load('oof_pointnet_clin.npy').mean(0),'1D CNN + clinical':np.load('oof_cnn_clin.npy').mean(0)}
fig,ax=plt.subplots(1,4,figsize=(15,4.4),sharey=True,facecolor='#fcfcfb')
for a,(k,p) in zip(ax,P.items()):
    a.set_facecolor('#fcfcfb'); a.scatter(p,np.log1p(f.y_min),s=30,color='#2a78d6',edgecolor='#fcfcfb',lw=1.5)
    T=[0,2,5,10,20,40]; lt=np.log1p(T); a.plot([0,lt[-1]],[0,lt[-1]],color='#999',lw=1)
    a.set_xticks(lt,[str(t) for t in T]); a.set_yticks(lt,[str(t) for t in T]); a.set_xlim(0,lt[-1]); a.set_ylim(0,lt[-1]); a.set_aspect('equal')
    a.set_title(f'{k}\npooled ρ = {spearmanr(p,f.y)[0]:.2f}',fontsize=10,color='#0b0b0b')
    a.set_xlabel('Predicted (min, log scale)',color='#52514e')
    for k2,s in a.spines.items():
        if k2 in('top','right'): s.set_visible(False)
ax[0].set_ylabel('Actual puncture to IC (min, log scale)',color='#52514e')
plt.tight_layout(); plt.savefig(f'{OUT}/plots/predicted_vs_actual.png',dpi=150)
print(r[['label','spearman','spearman_sd','mae_min','mae_min_sd']].iloc[::-1].round(2).to_string())
