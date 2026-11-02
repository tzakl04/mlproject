"""Extend the cohort with the new lab-processed subjects (CompleteStarDatabase/new, Nov 2026).
Clinical fields from cohort_flags.csv (STAR), morphology table from the lab's allsubs ImageTable,
operator unknown for the new subjects. Writes cohort_ext.csv, paths_ext.npy, ids_ext.json."""
import json, numpy as np, pandas as pd, scipy.io as sio, os, shutil, glob
D = '/home/claude/data'; W = '/home/claude/work'; NS = '/home/claude/newsubs'
UP = "/mnt/user-data/uploads/OneDrive - University of Virginia/Ullah, Tahmid Tahmid (mbp7jb)'s files - AI - segmentation/03_ModelOutputs/CompleteStarDatabase/new"
old = pd.read_csv(f'{W}/cohort102.csv')
flags = pd.read_csv(f'{NS}/cohort_flags.csv')
A = pd.read_excel(f'{NS}/ImageTableGS3506OperatorEngQ90allsubs.xlsx'); A.columns = [c.replace(' ', '_') for c in A.columns]
new_ids = json.load(open(f'{NS}/new_ids.json'))['infolder']
# copy staged files into the data folder (container-local, never OneDrive)
for i in new_ids:
    os.makedirs(f'{D}/{i}', exist_ok=True)
    for fn in glob.glob(f'{UP}/{i}/*.mat'):
        if not os.path.exists(f'{D}/{i}/{os.path.basename(fn)}'): shutil.copy(fn, f'{D}/{i}/')
c = flags[flags.STAR_ID.isin(new_ids)].copy()
c = c[c.Approach.isin(['Femoral', 'Radial']) & c.Location.eq('Anterior') & c['Puncture to IC access'].notna()]
c['Operator'] = 'Unknown'; c['decision'] = 'include (new batch)'
def route(appr, ves):
    side = 'L' if str(ves).strip().startswith('L') else 'R'
    return (0 if str(appr).lower().startswith('femoral') else 2) + (1 if side == 'L' else 0), side
def load(i, seg, r):
    m = sio.loadmat(f'{D}/{i}/smoothpathssubmmtrunc{seg}ant.mat', squeeze_me=True, struct_as_record=False); s = m['smoothPaths'][r]
    return s.name, np.asarray(s.path, float)
rows = []; paths = dict(np.load(f'{W}/paths.npy', allow_pickle=True).item())
for _, x in c.iterrows():
    i = x.STAR_ID; r, side = route(x.Approach, x['Vessel 1']); rec = dict(STAR_ID=i, route=r, side=side)
    for seg in ['arch', 'neck', 'archneck']:
        try:
            n, p = load(i, seg, r); rec[f'{seg}_name'] = n; rec[f'{seg}_n'] = len(p) if p.ndim == 2 else 0
            if p.ndim == 2 and len(p) > 200: paths[f'{i}_{seg}'] = p
        except Exception as e: rec[f'{seg}_err'] = str(e)[:60]
    rows.append(rec)
r = pd.DataFrame(rows); print(r.describe()); print('errors:', r.filter(like='err').notna().sum().to_dict() if r.filter(like='err').shape[1] else 'none')
bad = r[~(r.arch_n >= 200)].STAR_ID.tolist(); print('no usable arch path:', bad); print('neck/archneck missing:', r[~(r.neck_n >= 200) | ~(r.archneck_n >= 200)].STAR_ID.tolist())
cn = c.merge(r, on='STAR_ID'); cn = cn[~cn.STAR_ID.isin(bad)]
ext = pd.concat([old.assign(batch='old'), cn.assign(batch='new')], ignore_index=True)
ext.to_csv(f'{W}/cohort_ext.csv', index=False); np.save(f'{W}/paths_ext.npy', paths, allow_pickle=True)
json.dump([int(i) for i in ext.STAR_ID], open(f'{W}/ids_ext.json', 'w'))
print('cohort', len(ext), ext.batch.value_counts().to_dict(), ext.Approach.value_counts().to_dict(), ext.side.value_counts().to_dict())
