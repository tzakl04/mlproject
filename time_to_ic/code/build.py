import json, numpy as np, pandas as pd, scipy.io as sio, os
D='/home/claude/data'
c=pd.read_csv('/mnt/project-files/data-review/cohort_flags.csv')
b=pd.read_excel('/mnt/user-data/uploads/Kellogg Lab/AllFinalTableNewDampingQ90submm0826.xlsx')
ids=[int(i) for i in json.load(open('/home/claude/work/ids.json'))]
c=c[c.STAR_ID.isin(ids)&(c.STAR_ID!=309)].copy()
bb=b.drop_duplicates('SubjectNum').set_index('SubjectNum')
for i in c.index[c.Approach.isna()]:
    sid=c.at[i,'STAR_ID']; c.at[i,'Approach']=bb.at[sid,'Approach']; c.at[i,'Vessel 1']=bb.at[sid,'Vessel']+' (BE file)'
    dm={1:'Easy',2:'Average',3:'Challenging',4:'Very difficult',5:'Impossible'}
    c.at[i,'Arch difficulty']=dm[bb.at[sid,'ArchDifficulty']]; c.at[i,'Neck Difficulty']=dm[bb.at[sid,'NeckDifficulty']]
op=b.drop_duplicates('SubjectNum').set_index('SubjectNum')['Operator']
c['Operator']=c.STAR_ID.map(op).fillna('Unknown')
def route(appr, ves):
    side = 'L' if str(ves).strip().startswith('L') else 'R'
    a = str(appr).lower()
    base = 0 if a.startswith('femoral') else 2   # first route for crossovers
    return base + (1 if side=='L' else 0), side
def load(i, seg, r):
    m=sio.loadmat(f'{D}/{i}/smoothpathssubmmtrunc{seg}ant.mat',squeeze_me=True,struct_as_record=False)
    s=m['smoothPaths'][r]
    return s.name, np.asarray(s.path,float)
rows=[]; paths={}
for _,x in c.iterrows():
    i=x.STAR_ID; r,side=route(x.Approach,x['Vessel 1'])
    rec=dict(STAR_ID=i, route=r, side=side)
    for seg in ['arch','neck','archneck']:
        try:
            n,p=load(i,seg,r); rec[f'{seg}_name']=n; rec[f'{seg}_n']=len(p) if p.ndim==2 else 0
            if p.ndim==2 and len(p)>2: paths[(i,seg)]=p
        except Exception as e: rec[f'{seg}_err']=str(e)[:60]
    rows.append(rec)
r=pd.DataFrame(rows); print(r.head()); print(r.describe())
print(r[[k for k in r if 'err' in k]].notna().sum() if any('err' in k for k in r) else 'no errors')
print('missing archneck:', r[r.archneck_n<3].STAR_ID.tolist() if 'archneck_n' in r else '')
np.save('/home/claude/work/paths.npy', {f'{k[0]}_{k[1]}':v for k,v in paths.items()}, allow_pickle=True)
c.merge(r,on='STAR_ID').to_csv('/home/claude/work/cohort102.csv',index=False)
print(c.Operator.value_counts())
