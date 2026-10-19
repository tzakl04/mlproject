import sys, json, time; sys.path.insert(0,'/home/claude/work/src')
import numpy as np, pandas as pd, torch, torch.nn as nn
from cv import score
torch.set_num_threads(4)
arch=sys.argv[1]; use_clin=sys.argv[2]=='clin'; reps=int(sys.argv[3]) if len(sys.argv)>3 else 5
f=pd.read_csv('/home/claude/work/features.csv'); F=json.load(open('/home/claude/work/folds.json'))
Z=np.load('/home/claude/work/dl_inputs.npz'); prof,cloud,logL=Z['prof'],Z['cloud'],Z['logL']
clin_cols=['radial','left','age','male','op_Ryan','op_Min','op_Avery','op_Unknown']
C=f[clin_cols].astype(float).values; C[np.isnan(C)]=np.nanmedian(C,0)[np.where(np.isnan(C))[1]]
y=f.y.values.astype(np.float32)
class CNN1D(nn.Module):
    def __init__(s,cin=3):
        super().__init__()
        def blk(i,o): return nn.Sequential(nn.Conv1d(i,o,7,padding=3),nn.BatchNorm1d(o),nn.ReLU(),nn.MaxPool1d(2))
        s.f=nn.Sequential(blk(cin,16),blk(16,32),blk(32,32),blk(32,32),nn.Dropout(0.2))
        s.out=2*32+1
    def forward(s,x,l): h=s.f(x); return torch.cat([h.mean(2),h.amax(2),l[:,None]],1)
class PointNet(nn.Module):
    def __init__(s,cin=4):
        super().__init__()
        s.f=nn.Sequential(nn.Conv1d(cin,32,1),nn.BatchNorm1d(32),nn.ReLU(),nn.Conv1d(32,64,1),nn.BatchNorm1d(64),nn.ReLU(),nn.Conv1d(64,64,1),nn.BatchNorm1d(64),nn.ReLU())
        s.out=2*64+1
    def forward(s,x,l): h=s.f(x); return torch.cat([h.amax(2),h.mean(2),l[:,None]],1)
class Net(nn.Module):
    def __init__(s,enc,nclin):
        super().__init__(); s.enc=enc; s.head=nn.Sequential(nn.Linear(enc.out,32),nn.ReLU(),nn.Dropout(0.3),nn.Linear(32,1))
        s.lin=nn.Linear(nclin,1) if nclin else None
    def forward(s,x,l,c):
        o=s.head(s.enc(x,l))[:,0]
        if s.lin is not None: o=o+s.lin(c)[:,0]
        return o
def get_x(idx,aug_idx):
    if arch=='cnn': return torch.tensor(prof[idx,aug_idx])
    return torch.tensor(cloud[idx,aug_idx]).permute(0,2,1).contiguous()
def run_fold(tr,te,seed):
    torch.manual_seed(seed); rng=np.random.default_rng(seed)
    mu,sd=y[tr].mean(),y[tr].std(); cm,cs=C[tr].mean(0),C[tr].std(0)+1e-6
    Cn=torch.tensor((C-cm)/cs,dtype=torch.float32); yt=torch.tensor((y-mu)/sd)
    net=Net(CNN1D() if arch=='cnn' else PointNet(), C.shape[1] if use_clin else 0)
    opt=torch.optim.AdamW(net.parameters(),lr=2e-3,weight_decay=1e-2)
    E=120; sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,E)
    tri=np.where(tr)[0]; lossf=nn.HuberLoss(delta=1.0)
    for ep in range(E):
        net.train(); perm=rng.permutation(tri)
        for bstart in range(0,len(perm),16):
            bi=perm[bstart:bstart+16]
            if len(bi)<2: continue
            ai=rng.integers(1,prof.shape[1],len(bi))
            x=get_x(bi,ai); l=torch.tensor(logL[bi,ai])
            loss=lossf(net(x,l,Cn[bi]),yt[bi]); opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    net.eval(); tei=np.where(te)[0]
    with torch.no_grad():
        # test-time: original + 8 augmented views averaged
        ps=[net(get_x(tei,np.full(len(tei),a)),torch.tensor(logL[tei,a]),Cn[tei]).numpy() for a in [0]+list(range(1,9))]
    return np.mean(ps,0)*sd+mu
res=[]; preds=[]; t0=time.time()
for rep,fold in enumerate(F[:reps]):
    fold=np.array(fold); pred=np.zeros(len(f))
    for j in range(5):
        pred[fold==j]=run_fold(fold!=j,fold==j,seed=rep*10+j)
    s=score(y,pred,f.y_min.values); res.append(s); preds.append(pred)
    print(arch,sys.argv[2],'rep',rep,{k:round(v,3) for k,v in s.items()},f'{time.time()-t0:.0f}s',flush=True)
d=pd.DataFrame(res)
out=dict(features=f'{arch}{" +clinical" if use_clin else " only"}',model='DL',spearman=d.spearman.mean(),spearman_sd=d.spearman.std(),mae_min=d.mae_min.mean(),mae_min_sd=d.mae_min.std(),mae_log=d.mae_log.mean())
print(out); json.dump(out,open(f'/home/claude/work/res_{arch}_{sys.argv[2]}.json','w'))
np.save(f'/home/claude/work/oof_{arch}_{sys.argv[2]}.npy',np.array(preds))
