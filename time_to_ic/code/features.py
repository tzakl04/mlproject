import sys; sys.path.insert(0,'/home/claude/work/src')
import numpy as np, pandas as pd
from geom import metrics, circumR, torsion, arclen
P=np.load('/home/claude/work/paths.npy',allow_pickle=True).item()
c=pd.read_csv('/home/claude/work/cohort102.csv')
b=pd.read_excel('/mnt/user-data/uploads/Kellogg Lab/AllFinalTableNewDampingQ90submm0826.xlsx').drop_duplicates('SubjectNum').set_index('SubjectNum')
f=pd.DataFrame({'STAR_ID':c.STAR_ID})
f['y_min']=c['Puncture to IC access'].values; f['y']=np.log1p(f.y_min)
f['group']=f.STAR_ID.replace({346:345,359:358})
# clinical
f['radial']=c.Approach.str.lower().str.startswith('radial').astype(int).values
f['left']=(c.side=='L').astype(int).values
f['age']=pd.to_numeric(c.Age,errors='coerce').values
f['male']=(c.Sex=='male').astype(int).values
for o in ['Ryan Kellogg','Min Park','Avery Evans','Unknown']: f['op_'+o.split()[0]]=(c.Operator==o).astype(int).values
dm={'Easy':1,'Average':2,'Challenging':3,'Very difficult':4,'Impossible':5}
f['arch_diff']=c['Arch difficulty'].map(dm).values; f['neck_diff']=c['Neck Difficulty'].map(dm).values
# lab table morphology (route-matched, from AllFinalTable)
tcols=[k for k in b.columns if any(k.startswith(p) for p in ['PL','TIG','curv','eng','vessRad']) and not k.startswith('engSim') and not k.startswith('TIGSim') and not k.endswith('ICA') and not k.endswith('Total')]
for k in tcols: f['tab_'+k]=f.STAR_ID.map(b[k])
# hand-crafted from current lab paths
for seg in ['arch','neck','archneck']:
    for i,sid in enumerate(f.STAR_ID):
        p=P[f'{sid}_{seg}']; m=metrics(p)
        R=circumR(p[:,:3]); k=1/R; k=k[np.isfinite(k)]
        # curvature on 1 mm scale (smooth noise): resample every 10 pts
        q=p[::10]; k1=1/circumR(q[:,:3]); k1=k1[np.isfinite(k1)]
        tor=np.abs(torsion(q[:,:3]))
        feats=dict(PL=m['PL'],TIG=m['TIG'],TIG2D=m['TIG2D'],eng=m['eng'],engG=m['engG'],
                   rad_mean=p[:,3].mean(),rad_min=np.percentile(p[:,3],5),
                   k1_mean=k1.mean(),k1_p90=np.percentile(k1,90),k1_max=k1.max(),
                   bends=(k1>0.1).mean(),tor_mean=tor.mean(),minR1=1/k1.max())
        for kk,v in feats.items(): f.at[i,f'hc_{seg}_{kk}']=v
f.to_csv('/home/claude/work/features.csv',index=False)
print(f.shape); print(f.filter(like='hc_archneck').describe().T.round(3))
