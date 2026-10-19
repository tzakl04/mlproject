# Predicting time to IC access from vessel morphology

Can the shape of the access route (aortic arch and carotid neck centerlines) predict
puncture-to-ICA time in thrombectomy better than operator and approach alone?

Cohort: 101 STAR cases with a lab-processed centerline for the treated route (femoral
or radial, left or right). Target: log(1 + minutes to IC). Crossovers excluded.

**No patient data is in this repository.** Centerline coordinates, the STAR table
and the per-case feature table live on the lab's OneDrive and in a local working
folder. The scripts expect them under `/home/claude/data` and `/home/claude/work`
(see `code/build.py`); only aggregate results and plots are committed.

## Pipeline (`code/`)

| Script | What it does |
| --- | --- |
| `build.py` | Picks the treated route per case from the lab's `smoothpathssubmmtrunc*ant.mat` files, fills approach and operator from the AllFinalTable file. |
| `geom.py` | Centerline metrics: path length, tortuosity (TIG), 3-point curvature, bending energy (90th percentile of the ±50-point windowed turn-angle² sum), torsion. |
| `validate.py` | Checks recomputed metrics against the lab's AllFinalTable values. |
| `features.py` | Builds the per-case feature table: clinical, lab-table metrics, hand-crafted per segment. |
| `cv.py` | Stratified group 5-fold CV × 5 repeats (same-patient pairs kept together); Spearman and MAE scoring. |
| `baselines.py` | Ridge and random-forest baselines on clinical, lab-table and hand-crafted feature sets. |
| `prep_dl.py`, `train_dl.py` | Round 1: 1D CNN on curvature/torsion/radius profiles and PointNet on 3D coordinates, with ±15° rotation, scaling, jitter and resampling augmentation. |
| `prep_dl2.py`, `pretrain.py`, `dl2.py` | Round 2: heavier augmentation (random warps, crops, smoothing), contrastive pretraining on all 811 unlabeled routes (every patient × 4 routes × 2 segments), residual-on-clinical target, mixup, seed ensembles, config sweep. |
| `plots.py` | Results table and figures. |

## Results

See `results/results_table.csv` and `results/plots/`. Round 1 headline: clinical
ridge ρ = 0.52, MAE 5.15 min; best morphology model (clinical + arch features)
ρ = 0.52, MAE 5.01 min, not significantly better; shape-only models ρ ≤ 0.15.
Round 2 sweep results are in `results/sweep_round2.csv`.
