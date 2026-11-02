import sys; sys.path.insert(0,'/home/claude/work/src')
import numpy as np, pandas as pd
from geom import circumR, torsion, arclen
P=np.load('/home/claude/work/paths.npy',allow_pickle=True).item()
f=pd.read_csv('/home/claude/work/features.csv')
rng=np.random.default_rng(0)
N1=256; N2=512; K=40
def resample(p,n,offset=0.0,stretch=1.0):
    s=arclen(p[:,:3]); L=s[-1]
    a=np.clip(offset,0,0.05*L); b=L-np.clip(rng.uniform(0,0.03*L) if stretch!=1.0 else 0,0,0.05*L)
    t=np.linspace(a,b,n)
    return np.stack([np.interp(t,s,p[:,j]) for j in range(4)],1)
def rotmat(deg):
    ax=rng.normal(size=3); ax/=np.linalg.norm(ax); th=np.deg2rad(rng.uniform(-deg,deg))
    K_=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(th)*K_+(1-np.cos(th))*K_@K_
def profile(q):
    xyz=q[:,:3]; k=1/circumR(xyz); k[~np.isfinite(k)]=0; k=np.nan_to_num(k)
    tor=torsion(xyz); tor=np.clip(np.nan_to_num(tor),-1,1)
    k=np.clip(k,0,1.0)
    return np.stack([k*10, tor*5, q[:,3]/3.0],0).astype(np.float32)  # rough unit scaling
def aug_path(p, aug):
    q=p.copy()
    if aug:
        s=rng.uniform(0.92,1.08); R=rotmat(15)
        c0=q[0,:3]; q[:,:3]=(q[:,:3]-c0)@R.T*s+c0; q[:,3]*=rng.uniform(0.95,1.05)
        q[:,:3]+=rng.normal(0,0.15,q[:,:3].shape)  # jitter (mm) on 0.1 mm path, smoothed below
        # light smoothing so jitter doesn't explode curvature
        ker=np.ones(15)/15
        for j in range(3): q[:,j]=np.convolve(np.pad(q[:,j],7,mode='edge'),ker,'valid')
        off=rng.uniform(0,0.03)
    else: off=0.0
    if not aug:
        ker=np.ones(15)/15
        for j in range(3): q[:,j]=np.convolve(np.pad(q[:,j],7,mode='edge'),ker,'valid')
    return q, off
prof=np.zeros((len(f),K+1,3,N1),np.float32); cloud=np.zeros((len(f),K+1,N2,4),np.float32); logL=np.zeros((len(f),K+1),np.float32)
for i,sid in enumerate(f.STAR_ID):
    p=P[f'{sid}_archneck']
    for a in range(K+1):
        q,off=aug_path(p,a>0)
        L=arclen(q[:,:3])[-1]
        r1=resample(q,N1,offset=off*L,stretch=1.0 if a==0 else 0.97)
        prof[i,a]=profile(r1); logL[i,a]=np.log(L/300)
        r2=resample(q,N2,offset=off*L,stretch=1.0 if a==0 else 0.97)
        xyz=(r2[:,:3]-r2[0,:3])/100.0
        cloud[i,a,:,:3]=xyz; cloud[i,a,:,3]=r2[:,3]/3.0
np.savez_compressed('/home/claude/work/dl_inputs.npz',prof=prof,cloud=cloud,logL=logL)
print(prof.shape, prof[:,0].mean((0,2)), prof[:,0].std((0,2)), cloud.shape)
