import numpy as np
def circumR(p):
    a=p[:-2]; b=p[1:-1]; c=p[2:]
    ab=np.linalg.norm(a-b,axis=1); bc=np.linalg.norm(b-c,axis=1); ca=np.linalg.norm(c-a,axis=1)
    area=0.5*np.linalg.norm(np.cross(b-a,c-a),axis=1)
    with np.errstate(divide='ignore',invalid='ignore'):
        R=ab*bc*ca/(4*area)
    R[~np.isfinite(R)]=np.inf
    return np.concatenate([[np.nan],R,[np.nan]])
def arclen(p):
    return np.concatenate([[0],np.cumsum(np.linalg.norm(np.diff(p,axis=0),axis=1))])
def turn_angle(p):
    d=np.diff(p[:,:3],axis=0)
    v1,v2=d[:-1],d[1:]
    cos=np.sum(v1*v2,1)/np.linalg.norm(v1,axis=1)/np.linalg.norm(v2,axis=1)
    ang=np.real(np.arccos(np.clip(cos,-1,1)))
    e=np.zeros(len(p)); e[1:-1]=ang
    return e
def energy_q90(p,win=50):
    e=turn_angle(p)**2
    cs=np.concatenate([[0],np.cumsum(e)])
    n=len(e); i=np.arange(n)
    lo=np.maximum(i-win,0); hi=np.minimum(i+win,n-1)
    E=cs[hi+1]-cs[lo]
    return np.quantile(E,0.9), e.sum()
def torsion(p):
    # discrete torsion via binormal rotation (rad per mm)
    d=np.diff(p,axis=0); t=d/np.linalg.norm(d,axis=1,keepdims=True)
    b=np.cross(t[:-1],t[1:]); bn=np.linalg.norm(b,axis=1,keepdims=True); b=b/np.where(bn<1e-12,np.inf,bn)
    tw=np.einsum('ij,ij->i',np.cross(b[:-1],b[1:]),t[1:-1])
    ang=np.arcsin(np.clip(tw,-1,1))
    ds=np.linalg.norm(d,axis=1)[1:-1]
    out=np.zeros(len(p)); out[2:-1]=ang/ds
    return out
def metrics(path,win=50):
    xyz=path[:,:3]
    L=arclen(xyz); R=circumR(xyz)
    eng,engG=energy_q90(path,win)
    chord=np.linalg.norm(xyz[0]-xyz[-1]); chord2=np.linalg.norm(xyz[0,:2]-xyz[-1,:2])
    L2=arclen(xyz[:,:2])[-1]
    return dict(PL=L[-1], TIG=L[-1]/chord, TIG2D=L2/chord2, curv=np.nanmin(R), vessRad=path[:,3].mean(), eng=eng, engG=engG)
