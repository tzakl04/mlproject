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
| `prep_seq.py`, `dl_seq.py`, `dl_seq2.py`, `pretrain_seq.py` | Round 4: each path as a sequence of direction changes (unit tangents in a canonical frame + rotation-invariant turn vectors in a parallel-transport frame + radius); 1D CNN, bidirectional GRU and Transformer encoders, contrastive pretraining on the same corpus, optional auxiliary difficulty-rating heads and pairwise rank loss. |
| `plots.py` | Results table and figures. |

## Results

See `results/results_table.csv` and `results/plots/`. Round 1 headline: clinical
ridge ρ = 0.52, MAE 5.15 min; best morphology model (clinical + arch features)
ρ = 0.52, MAE 5.01 min, not significantly better; shape-only models ρ ≤ 0.15.
Round 2 (`results/sweep_round2.csv`, `results/results_round2_confirm.csv`,
`results/plots/round2_sweep_confirm.png`): the arch segment alone, a contrastively
pretrained encoder, and predicting the residual on top of the clinical ridge each help.
The best sweep configs reach ρ 0.58-0.59 on the sweep folds, but on fresh folds (seeds
200-204) the pretrained arch CNN is ρ 0.548 vs 0.505 for clinical ridge (Δρ +0.044,
95% CI -0.031 to +0.124; ΔMAE -0.22 min, CI -0.64 to +0.14) and the CNN+PointNet
ensemble ρ 0.558 (Δρ +0.054, CI -0.016 to +0.129). A consistent but small gain whose
confidence interval still includes zero at n = 101. Heavier augmentation mattered less
than pretraining: without the unlabeled-route pretraining the CNN matches the baseline.

Round 3 (`results/round3/`): richer feature models (elastic net, PLS, kernel ridge,
boosting, forest, two-stage), frozen pretrained embeddings, a fused arch+neck encoder,
a wider encoder pretrained on 1,292 routes, a per-operator gain, slow-access
classification (AUC) and a within-operator analysis. None beats the round-2 model.
The one sizeable, significant finding: within Min Park's 36 cases the deep ensemble's
morphology adjustment correlates with the clinical residual at rho 0.55 (CI 0.28-0.73,
permutation p = 0.002); within Ryan Kellogg's 45 cases (median 5 min) rho 0.18, n.s.
Round 4 (`results/round4/`): the path encoded directly as a sequence of direction
changes (turn vectors) and fed to a CNN, GRU and Transformer. On sweep folds the pretrained
turn-vector CNN reaches rho 0.548 and the GRU 0.540, the Transformer 0.530; tangent-vector and
mixed channels, difficulty-rating auxiliary heads and a rank loss were all worse. On fresh
folds the turn-vector CNN is rho 0.504 (no better than clinical) and the GRU 0.520. Adding the
GRU to the round-2 ensemble gives rho 0.568, MAE 4.90 min (d_rho +0.064, CI +0.001 to +0.139;
d_MAE -0.39 min, CI -0.72 to -0.08 vs clinical ridge), the first interval that excludes zero,
though the ensemble was chosen after seeing the fresh-fold results.
Full report: see the project thread. Next step is more labelled centerlines
(112 anterior cases with time to IC have no processed path yet).
