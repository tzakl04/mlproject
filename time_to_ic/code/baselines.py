import sys; sys.path.insert(0,'/home/claude/work/src')
import numpy as np, pandas as pd, json
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from cv import score
f=pd.read_csv('/home/claude/work/features.csv'); F=json.load(open('/home/claude/work/folds.json'))
for k in [c for c in f if 'TIG2D' in c]: f[k]=np.log(f[k])
clin=['radial','left','age','male','op_Ryan','op_Min','op_Avery','op_Unknown']
tab=[c for c in f if c.startswith('tab_')]
hc=[c for c in f if c.startswith('hc_')]
hc_core=[f'hc_{s}_{k}' for s in ['arch','neck','archneck'] for k in ['PL','TIG','k1_mean','k1_p90','rad_mean','rad_min','tor_mean']]
sets={'null':[], 'clinical':clin, 'clinical+lab table':clin+tab, 'lab table only':tab,
      'clinical+hand-crafted':clin+hc_core, 'hand-crafted only':hc_core,
      'clinical+arch hand-crafted':clin+[c for c in hc_core if c.startswith('hc_arch_')],
      'clinical+difficulty (post-hoc ref)':clin+['arch_diff','neck_diff']}
models={'ridge':lambda: make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),RidgeCV(alphas=np.logspace(-2,3,30))),
        'rf':lambda: make_pipeline(SimpleImputer(strategy='median'),RandomForestRegressor(300,min_samples_leaf=5,max_features=0.5,random_state=0,n_jobs=4))}
rows=[]; oof={}
for name,cols in sets.items():
  for mn,mk in models.items():
    if name=='null' and mn=='rf': continue
    res=[]; preds=[]
    for rep,fold in enumerate(F):
        fold=np.array(fold); pred=np.zeros(len(f))
        for j in range(5):
            tr,te=fold!=j,fold==j
            if not cols: pred[te]=f.y[tr].mean(); continue
            m=mk(); m.fit(f.loc[tr,cols],f.y[tr]); pred[te]=m.predict(f.loc[te,cols])
        res.append(score(f.y.values,pred,f.y_min.values)); preds.append(pred)
    d=pd.DataFrame(res); oof[f'{name}|{mn}']=np.mean(preds,0)
    rows.append(dict(features=name,model=mn,n_feat=len(cols),spearman=d.spearman.mean(),spearman_sd=d.spearman.std(),mae_min=d.mae_min.mean(),mae_min_sd=d.mae_min.std(),mae_log=d.mae_log.mean()))
r=pd.DataFrame(rows); print(r.round(3).to_string()); r.to_csv('/home/claude/work/results_baselines.csv',index=False)
pd.DataFrame(oof).assign(STAR_ID=f.STAR_ID,y=f.y).to_csv('/home/claude/work/oof_baselines.csv',index=False)
