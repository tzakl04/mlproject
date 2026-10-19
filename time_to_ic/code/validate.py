import sys; sys.path.insert(0,'/home/claude/work/src')
import numpy as np, pandas as pd
from geom import metrics
P=np.load('/home/claude/work/paths.npy',allow_pickle=True).item()
c=pd.read_csv('/home/claude/work/cohort102.csv')
b=pd.read_excel('/mnt/user-data/uploads/Kellogg Lab/AllFinalTableNewDampingQ90submm0826.xlsx').drop_duplicates('SubjectNum').set_index('SubjectNum')
seg={'arch':'Arch','neck':'Neck','archneck':'ArchNeck'}
rows=[]
for i in c.STAR_ID:
  for s,S in seg.items():
    m=metrics(P[f'{i}_{s}'])
    for k,v in m.items(): rows.append(dict(id=i,seg=S,m=k,mine=v,lab=b.loc[i,f'{k}{S}'] if (i in b.index and f'{k}{S}' in b) else np.nan))
r=pd.DataFrame(rows); r.to_csv('/home/claude/work/validate.csv',index=False)
r['rel']=(r.mine-r.lab).abs()/r.lab.abs()
g=r.dropna(subset=['lab']).groupby(['seg','m']).rel
print(pd.DataFrame({'median_rel_err':g.median(),'frac_within_1pct':g.apply(lambda x:(x<0.01).mean()),'n':g.size()}).round(4))
