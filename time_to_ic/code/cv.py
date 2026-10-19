import numpy as np, pandas as pd, json
from sklearn.model_selection import StratifiedGroupKFold
from scipy.stats import spearmanr
def make_folds(f, n_rep=5, k=5):
    yb=pd.qcut(f.y,5,labels=False,duplicates='drop').values
    out=[]
    for r in range(n_rep):
        sg=StratifiedGroupKFold(k,shuffle=True,random_state=100+r)
        fold=np.zeros(len(f),int)
        for j,(_,te) in enumerate(sg.split(f,yb,f.group)): fold[te]=j
        out.append(fold.tolist())
    return out
def score(y, pred, ymin):
    return dict(spearman=spearmanr(y,pred)[0], mae_log=np.mean(np.abs(y-pred)), mae_min=np.mean(np.abs(ymin-np.expm1(pred))))
def summarize(res):
    d=pd.DataFrame(res); return {k:(d[k].mean(),d[k].std()) for k in d}
