# Project Summary — Coronal Hole Features for Solar Wind Speed Forecasting

## Current organization note (2026-06-24)

The project root has been tidied without deleting files. Canonical data and notebooks remain in `data/` so existing notebook-relative paths still work. Execution logs moved to `logs/`. Superseded root-level copies moved to `archive/root_legacy/`. One-off scratch artifacts moved to `archive/scratch/`. Stale presentation figures moved to `archive/presentation_stale_figures/`. See `README.md` for the concise current layout.

Handoff document for a fresh agent/conversation. Written 2026-06-16. Read this before doing anything else in this project.

## Goal

Build HMI-magnetogram-derived coronal hole (CH) features and test whether they improve XGBoost solar wind speed forecasts, alongside an existing EUV-based CH feature set and a richly-engineered baseline dataset (`data/4x3.csv`).

## Directory layout (as of this writing)

```
D:\kaijiel3\Downloads\hmi\
├── PROJECT_SUMMARY.md              <- this file
├── hmi_processing.ipynb            <- ROOT COPY, superseded — see data/ for canonical files
├── hmi_2010-05-01_to_2010-12-31_6h.csv   <- ROOT COPY, duplicate of data/ version
├── hmi_2014_6h.csv                 <- ROOT COPY, duplicate of data/ version
├── run_2014.py                     <- ROOT COPY, superseded by data/run_hmi_year.py
├── data/
│   ├── 4x3.csv                     <- BASE dataset (hourly, 2010-06 to 2024-06, 123,456 rows).
│   │                                   Col 1 = target `speed`. Other cols are pre-engineered
│   │                                   lag features: magn/temp/dens/speed/press at -26/-27/-28
│   │                                   days, a 2x3 lat/lon CH grid (S[i,j]/D[i,j]) at -4 to -7
│   │                                   days, sunspot_number(_change).
│   ├── renamed_no_proxy.csv        <- 193A/211A EUV-based CH features (global + same 2x3 grid),
│   │                                   6-hourly, 2010-05-13 to 2019-12-31, 13,229 rows, UNLAGGED.
│   ├── enhancements/
│   │   ├── hss_list.csv            <- High-speed-stream event catalog (start/end/peak), 2000-2024,
│   │   │                               519 events. CH-driven — used for physically-targeted eval.
│   │   ├── cme_list.csv            <- CME event catalog (different physical driver, ~negative control)
│   │   └── enhancement_list.csv    <- combined list with cme_flag
│   ├── hmi_2010-05-01_to_2010-12-31_6h.csv  <- HMI features, 2010 (911 rows). DONE.
│   ├── hmi_2014_6h.csv             <- HMI features, 2014 (1448 rows). DONE.
│   ├── hmi_2011_6h.csv             <- HMI features, 2011 (1433 rows of 1460 expected). DONE
│   │                                   (resumed from a 123-row checkpoint on 2026-06-16).
│   ├── hmi_2012_6h.csv             <- HMI features, 2012 (1438 rows of 1464). DONE (resumed).
│   ├── hmi_2013_6h.csv             <- HMI features, 2013 (1414 rows of 1460). DONE (resumed).
│   ├── hmi_2015_6h.csv             <- HMI features, 2015 (1445 rows of 1460). DONE.
│   ├── hmi_2016_6h.csv             <- HMI features, 2016 (1405 rows of 1464). DONE.
│   ├── hmi_2017_6h.csv             <- HMI features, 2017 (1440 rows of 1460). DONE.
│   ├── hmi_2018_6h.csv             <- HMI features, 2018 (1439 rows of 1460). DONE.
│   ├── hmi_2019_6h.csv             <- HMI features, 2019 (1430 rows of 1460). DONE.
│   ├── hmi_2020_6h.csv             <- HMI features (OLD global schema), 2020. Kept as reference.
│   ├── hmi_{2010..2020}_6h_grid.csv  <- HMI features (NEW per-CH grid schema, 2026-06-19), all
│   │                                   11 years, 15,213 rows total. R2_{i}_{j}/skew_{i}_{j} for
│   │                                   i=1..4 (lat band)/j=1..3 (lon band), plus n_ch_total/
│   │                                   n_ch_kept. Old hmi_{YEAR}_6h.csv files are untouched/kept.
│   ├── hmi_2010_2020_6h_grid.csv   <- Combined version of the 11 per-year grid CSVs above
│   │                                   (concat + sort by timestamp, no dedup needed since years
│   │                                   don't overlap). Same 15,213 rows/columns as the per-year
│   │                                   files combined, range 2010-05-13 -> 2020-12-31. Per-year
│   │                                   files are still kept too, this is purely a convenience
│   │                                   merge, not a replacement.
│   ├── run_hmi_year.py             <- CANONICAL, reusable, parameterized HMI-year processor.
│   │                                   Usage: python run_hmi_year.py <YEAR> [max_rows]  (e.g. 2021,
│   │                                   or `2014 8` to process only the first 8 timestamps for a
│   │                                   quick test). Writes hmi_{YEAR}_6h_grid.csv (per-CH grid
│   │                                   schema, since 2026-06-19 -- see "Immediate next steps").
│   │                                   SUPPORTS RESUME: if OUT_CSV already exists, it loads
│   │                                   existing rows and only processes timestamps after the
│   │                                   last one present, instead of overwriting from scratch.
│   ├── solar_wind_boosting_pipeline.ipynb  <- the ML pipeline notebook (see below)
│   ├── stageB_cross_regime_results_full.csv      <- full cross-regime CV results (220 rows)
│   ├── stageB_cross_regime_hss_split_full.csv    <- same, split by HSS event (440 rows)
│   └── run_201{1,2,3}.log, run_201{1,2,3}_resume.log, run_201{5,6,7,8,9}.log, run_2020.log
│                                       <- per-year processing logs
└── presentation/
    ├── stage_AB_summary.tex        <- Beamer deck, rewritten 2026-06-16 for the final embargoed
    │                                   full-2010-2020 results. NOT compiled (user declined a
    │                                   LaTeX distro). No longer references the two PNGs below —
    │                                   those were extracted from the OLD 2010+2014 pilot notebook
    │                                   run and would now show stale/mismatched SHAP and
    │                                   cross-regime plots, so the image slides were dropped in
    │                                   favor of tables built from the current numbers.
    ├── fig_stageA_shap.png         <- STALE, orphaned (was from the old pilot run, unreferenced)
    └── fig_stageB_crossregime.png  <- STALE, orphaned (same)
```

**Cleanup note**: the four ROOT-level files listed above are leftovers from before `data/` existed. They are stale/duplicate; the `data/` versions are canonical. Safe to delete the root copies, but nobody has done so yet — confirm with the user first since deleting files wasn't explicitly requested.

## Environments

Anaconda is at `F:\Anaconda`. Relevant envs (NOT on PATH — always invoke via full path):

- **`F:\Anaconda\envs\py38\python.exe`** — has `sunpy` (2.1.0, OLD), `zarr`, `s3fs`, `astropy`, `reproject`, `requests`, `tqdm`, `pandas`/`numpy`. Used for all HMI/SPoCA data pipeline work. Note: sunpy 2.1.0 predates `SphericalScreen` and chokes on tz-aware ISO date strings — see Gotchas below.
- **`F:\Anaconda\envs\ch_sws_prediction\python.exe`** — the project's dedicated ML env. Has pandas 3.0, numpy 2.4, scikit-learn 1.8, **xgboost 3.2.0 (CUDA-enabled)**, shap 0.52, jupyter/nbconvert/ipykernel, matplotlib, **optuna 4.9.0 (added 2026-06-21, was missing before)**. Used for `solar_wind_boosting_pipeline.ipynb`.

**GPU**: machine has an NVIDIA RTX 4060 with working drivers (`nvidia-smi` works). The installed XGBoost build has `USE_CUDA: True`. The boosting notebook uses `device='cuda', tree_method='hist'` in `XGB_PARAMS`/`XGB_PARAMS_REGULARIZED`.

No LaTeX distribution is installed on this machine (user declined a MiKTeX install via winget).

## Data source facts (verified directly against S3, not assumed)

S3 bucket: `s3://gov-nasa-hdrl-data1/contrib/fdl-sdoml/fdl-sdoml-v2/`

| Zarr store | Years available | Resolution |
|---|---|---|
| `sdomlv2_hmi_small.zarr` | **2010 only** | 512x512 |
| `sdomlv2_hmi.zarr` | **2010-2020** (all 11 years confirmed) | 512x512 (same as "small"!) |
| `sdomlv2.zarr`, `sdomlv2_small.zarr` | similar pattern (multi-channel versions) | — |

Key realization: `sdomlv2_hmi.zarr` ("full" store) is NOT higher resolution than the "small" store for HMI Bz — both are 512x512. So there's no performance penalty to using the full store for years beyond 2010; it was just never explained why two stores exist, but it doesn't matter for this work.

**SPoCA coronal hole maps** (`https://spoca.oma.be/spoca4tap/rob_spoca_ch/ch_map/{YYYYMMDD_HHMMSS}.ch_map.fits`): directory-listing confirmed coverage from **2010-05-13** through **2025-12-31**, ~22,467 files at 6-hour cadence (00/06/12/18 UTC), ~98% complete (sparse genuine gaps are normal/expected, handled by treating HTTP 404 as a skip). CH maps are **integer-labeled regions** (e.g. values like 74, 107, 116 = distinct CH IDs), not binary masks — thresholding at `>0.5` still works correctly since 0=background and all real labels are positive integers.

## HMI feature definitions (what `run_hmi_year.py` currently computes — CURRENT, as of 2026-06-19)

Per timestamp, **per-individual-coronal-hole** features assigned to a 4 (lat) x 3 (lon) grid — see "Immediate next steps" below for full detail and rationale. Each qualifying CH (>= 10 pixels, identified via SPoCA's own integer CH-ID labels preserved through a nearest-neighbor reprojection) gets:

- `R2` = `2 * |0.5 - Φ+/(Φ+ + |Φ-|)|` — magnetic flux imbalance ratio (Φ+ = positive flux sum, Φ- = negative flux sum), same formula as before but now computed per-CH instead of pooled across the whole disk.
- `skew` = Fisher-Pearson skewness of that CH's Bz pixel values.

Each CH is assigned to one of 12 grid cells (`LAT_LINES=[-45,0,45]` x `LON_LINES=[-30,30]`) by its pixel-centroid lat/lon; if multiple CHs land in the same cell, the largest (by pixel count) wins. Output columns: `R2_{i}_{j}`, `skew_{i}_{j}` (i=1..4, j=1..3), `n_ch_total`, `n_ch_kept`.

**Superseded (kept as historical/comparison data, not deleted)**: the original definition computed four **global** features by pooling every CH-masked pixel on the disk together — `global_R2`, `global_energy` (`mean(Bz^2)`), `global_entropy` (Shannon entropy, 100-bin histogram), `global_variance` (`var(Bz)`) — with a whole-disk fallback if zero CH pixels were detected. Still present in the old `hmi_{YEAR}_6h.csv` files and in `hmi_processing.ipynb` (which is itself fully stale/superseded, predating even the canonical `run_hmi_year.py` — left untouched, not part of either feature generation). `global_R2` was by far the most useful of the four in modeling under the old scheme. A 6-cell "grid" variant (2x3, matching `renamed_no_proxy.csv`'s pattern) was scoped early on but never implemented under the old scheme — that's the `abs_lat_bounds`/`lon_bounds` leftover TODO scaffolding in `hmi_processing.ipynb`, now superseded by the actually-implemented 4x3 per-CH grid described above.

## The boosting model pipeline (`data/solar_wind_boosting_pipeline.ipynb`)

Two-stage design, GPU-trained XGBoost, chronological CV throughout (never randomly shuffled):

- **Stage A**: validates the lag-and-merge methodology at scale using `renamed_no_proxy.csv` (EUV features, ~10yr/123K rows). 4-day lag (a feature observed at `T_obs` predicts the target at `T_obs + 4 days`, matching the existing `-4` to `-28`-day convention already in `4x3.csv`). Merge is `pd.merge_asof` (forward-fill, 48h tolerance cap) onto the hourly target grid.
- **Stage B**: the actual question — does adding HMI features help? Only ~1.6yr of HMI data exists (2010 + 2014, two disjoint solar-cycle regimes: minimum vs. maximum), so this is treated as a **pilot**. Evaluated via (a) time-series CV within the combined window and (b) cross-regime holdout (train on one regime, test on the other, both directions) — the latter is the real generalization test.
- **HSS-event split**: results also broken out by whether the target hour falls inside a cataloged high-speed-stream event (`hss_list.csv`), since HSS is the CH-driven phenomenon these features should specifically help predict.

### Current verified results (after the regularization fix — see Gotchas)

- Stage A: EUV features rank highly by SHAP (best rank 7/138, confirms lag direction correct) but don't improve average CV RMSE/MAE over the baseline alone (likely redundant with `4x3.csv`'s own `-4` to `-7`-day CH grid features). Both clearly beat a naive 27-day-recurrence baseline.
- Stage B (regularized): HMI features improve **every single comparison** run — within-window CV, both cross-regime holdout directions, and all 4 HSS-event-split combinations. One direction's R² flips from negative (-0.018) to positive (+0.021) when HMI is added. `global_R2` does essentially all the work (SHAP rank 7/142); energy/entropy/variance are comparably uninformative (tied at rank ~105). HMI features are only weakly correlated with existing EUV CH-area features (|r| ≤ 0.32) — they carry distinct information.
- Despite the positive direction, absolute cross-regime R² is still mostly negative — generalizing across solar-cycle phase is intrinsically hard. **The natural next step is more HMI years** to turn this 2-point pilot into a statistically robust test — which is exactly what the in-progress/stopped work (2011-2013, 2015-2020) is for.

## Gotchas / bugs found and fixed along the way (don't re-introduce these)

1. **`sdomlv2_hmi_small.zarr` only has 2010** — don't use it for other years; use `sdomlv2_hmi.zarr/{year}` instead (same resolution, confirmed).
2. **tz-aware timestamps break old sunpy**: `obs_time.isoformat()` on a tz-aware pandas Timestamp produces a `+00:00`-suffixed string that crashes sunpy 2.1.0's WCS date parsing. Fix: `obs_time.tz_localize(None).isoformat()` before putting it in the FITS-style header dict.
3. **Off-disk pixel warnings**: `hpc_coords.transform_to(HeliographicStonyhurst)` spams `SunpyUserWarning` for off-disk pixels unless wrapped in `with SphericalScreen(sdoml_map.observer_coordinate):`. (Only matters for the never-implemented grid-feature code path in `hmi_processing.ipynb`.)
4. **Column name sanitization for XGBoost**: raw `4x3.csv`/`renamed_no_proxy.csv` column names contain `[`, `]`, `(`, `)`, `,`, `"` which XGBoost rejects. Use regex substitution, not `str.maketrans` (a maketrans length-mismatch bug was hit once — `re.sub(r'[\[\](),"]', '_', name)` is the working approach, see `sanitize_columns()` in the notebook).
5. **CSV round-trip loses datetime dtype**: reloading a checkpoint CSV via `pd.read_csv` without `parse_dates=['timestamp']` gives back strings, not Timestamps — mixing those with fresh Timestamp objects in the same column crashes `sort_values()`. Always pass `parse_dates=` when reloading a checkpoint.
6. **In-sample vs out-of-sample evaluation bug**: an early version of the HSS-event-split analysis evaluated `model.predict()` on the same data the model was trained on, giving deceptively tiny (meaningless) RMSE values. Fixed by collecting genuine out-of-fold (CV) or held-out (cross-regime test) predictions before doing any event-based slicing. `evaluate_cv()` returns `(metrics_df, oof_pred_series)` for exactly this reason — always use the oof series for any post-hoc slicing, never refit-and-predict-on-training-data.
7. **Overfitting in Stage B**: reusing Stage A's hyperparameters (300 trees, depth 6, no regularization) on Stage B's much smaller per-regime samples (5-9K rows) gave train R² of 0.98-0.996 with cross-regime test R² as low as -1.29 — classic overfitting, NOT primarily a target-distribution shift (regime means were only ~3% apart, verified directly). Fixed via `XGB_PARAMS_REGULARIZED` (depth 3, subsample/colsample_bytree=0.7, min_child_weight=10, reg_lambda=5, reg_alpha=1, early stopping with `early_stopping_rounds=30` on a chronological validation tail carved from the *end* of the training portion via `fit_model()`). This dropped train R² to a sane 0.25-0.75 and, notably, made Stage B's "HMI helps" conclusion *stronger and more consistent*, not weaker.
8. **Background jobs can die silently**: one long-running background CPU job died with no error, no traceback, and no completion notification partway through (cause never determined — possibly an external interruption). Mitigation in place: checkpointing every 25 rows in `run_hmi_year.py`, so a resume only needs to reprocess from the last checkpoint, not from scratch. If you restart processing and a job goes quiet for longer than expected, check `Get-Process python` and the log's `LastWriteTime` directly rather than assuming the harness will always notify on death.
9. **Read tool can show a stale cache for notebooks modified by an external process** (e.g., after running `nbconvert --execute` which rewrites the `.ipynb` outside the Read/Write/Edit tool chain). If `Read` says "file unchanged since your last Read" right after an nbconvert run, that's wrong — verify via `Get-Item ... | Select LastWriteTime` and extract content via a small Python/json script instead of trusting the Read tool's dedup logic in that situation.

## Immediate next steps (where the user paused)

**2026-06-16 update**: 2011/2012/2013 were resumed from their checkpoints (`run_hmi_year.py` now supports resume — see above) and ran to completion successfully, in parallel, with no exceptions. Then 2015-2020 were run fresh (6-way parallel, all from scratch, one harmless exception in 2020) and also completed successfully.

**All years 2010-2020 are now DONE** (HMI data only goes back to 2010 and the SDOML zarr store currently covers through 2020 — see "Data source facts" above). No years remain unprocessed within the currently-known data availability window; the original pilot's "more HMI years" ask is now satisfied for the full SDOML coverage range.

**Stage B re-run with full 2010-2020 data — DONE, and the conclusion flipped.** The 2-year pilot (2010+2014) claimed HMI features "improve every single comparison." With the full 11-year dataset that does NOT hold — see numbers below.

This required a code fix: `run_hmi_year`-derived gap-based regime detection (>30-day timestamp gaps = new regime) was a hack that only worked when years were sparse (2010, 2014 only). With 2010-2020 continuously covered, consecutive years have no real gap at the Dec 31 -> Jan 1 boundary, so it collapsed to one mega-regime. Fixed by switching `regime_id` to calendar year (`solar_wind_boosting_pipeline.ipynb` cell 11) — gives 12 regimes (2010 partial, 2011-2020 full, 2021 stray 139-row tail from the 4-day lag pushing late-Dec-2020 HMI obs into Jan-2021 targets).

**2026-06-16, second update — added a 2-month (60-day) train/test embargo to both CV schemes**, per explicit user request after discussing temporal-leakage risk at year boundaries (the lag-and-merge tolerance window can let feature data spill a few hours/days across the Dec31->Jan1 regime boundary):
- `evaluate_cv` (within-window `TimeSeriesSplit` CV): drops training rows within 60 days of each fold's test-start timestamp. New params: `timestamps=`, `embargo_days=EMBARGO_DAYS` (=60, defined in cell 1).
- `cross_regime_eval` (cross-regime holdout): drops training rows within 60 days of the test regime's date range — a no-op for non-adjacent year pairs, trims boundary rows for adjacent ones.
- Had to also exclude the degenerate 139-row "2021" stray-tail regime from the cross-regime loop (`MIN_REGIME_ROWS = 1000` filter in cell 13) — with the embargo, a regime that small can lose 100% of its rows as a training set when paired against an adjacent test year, crashing the model fit (`ValueError: Found array with 0 sample(s)`). 2021 stray tail is still included in the within-window CV (where it isn't a standalone train set).

**Final results (full 2010-2020 dataset, calendar-year regimes, 60-day embargo) — this is the current/correct state of the analysis:**

| Comparison | Control (no HMI) | Treatment (+HMI) |
|---|---|---|
| Within-window CV: RMSE / MAE / R² | 76.29 / 61.02 / 0.232 | 76.18 / 60.72 / **0.238** (slightly better) |
| Cross-regime, all 110 pairs: RMSE / MAE / R² | 87.40 / 69.04 / 0.003 | 87.96 / 69.59 / -0.011 (slightly worse) |
| Cross-regime, full-years-only (90 pairs, excl. 2010 partial): RMSE / MAE / R² | 87.46 / 68.78 / -0.016 | 88.00 / 69.29 / -0.028 (slightly worse) |
| Treatment beats control on RMSE | — | 69 / 110 pairs (63%) |
| Within-CV split by HSS event: RMSE during HSS / during quiet | 82.14 / 74.21 | 84.15 (**worse**) / 72.95 (**better**) |
| Cross-regime split by HSS event: RMSE during HSS / during quiet | 95.73 / 81.18 | 96.21 (worse) / 81.61 (worse) |

Key takeaways:
- The effect of adding HMI features is small and inconsistent — it nets out roughly neutral to slightly negative, nowhere near the original 2010+2014 pilot's "helps every single comparison." **Likely interpretation: the pilot's positive result was a small-sample artifact** — its two regimes happened to be an unusually favorable train/test pair, not representative of the full 11-year solar-cycle range.
- A genuinely new and counterintuitive finding from the within-window-CV HSS split: HMI features **hurt** prediction during actual high-speed-stream events (82.14 -> 84.15 RMSE) but **help** during quiet periods (74.21 -> 72.95 RMSE) — the opposite of the physically-motivated expectation (HSS is CH-driven, so HMI should help most exactly during HSS events). This reversal does not show up in the cross-regime HSS split (both event/non-event get slightly worse there), so treat it as a within-window-specific observation, not yet a robust finding.
- `global_R2` remains the standout HMI feature (SHAP rank 10/142); `energy`/`entropy`/`variance` remain comparably uninformative (ranks 50-128).

Full per-pair artifacts: `data/stageB_cross_regime_results_full.csv` (220 rows: 110 pairs x control/treatment) and `data/stageB_cross_regime_hss_split_full.csv` (440 rows: 110 pairs x control/treatment x hss/non-event).

**Done**: `presentation/stage_AB_summary.tex` has been rewritten to reflect this final (embargoed, full-11-year) result — see below. Still not compiled (no LaTeX distro installed; user previously declined MiKTeX via winget).

**2026-06-16, third update — within-window CV is now 5-fold with per-variant hyperparameter optimization (cross-regime holdout untouched, by explicit request).** Added `PARAM_DIST`, `random_param_search()`, and `tune_and_evaluate_cv()` to cell 5 — a single-level random search (15 candidates, each scored by its own embargoed n-fold CV, winning candidate's own fold metrics/oof predictions reported directly — same folds inform selection and reporting, a deliberate speed/rigor tradeoff vs. nested CV). Stage A's `XGB_PARAMS` (depth 6, 300 trees, no regularization, no early stopping) and Stage B's `XGB_PARAMS_REGULARIZED` (hand-tuned depth 3, etc.) are no longer used for within-window CV — both `XGB_PARAMS`/`XGB_PARAMS_REGULARIZED` constants are kept only because the (unchanged) cross-regime cell still references `XGB_PARAMS_REGULARIZED`. Stage A's within-window CV also gained early stopping for the first time (it had none before).

**Tuned within-window CV results** (5-fold, embargoed, GPU XGBoost, search space: max_depth/learning_rate/subsample/colsample_bytree/min_child_weight/reg_lambda/reg_alpha):

| | RMSE | MAE | R² | Best max_depth / learning_rate |
|---|---|---|---|---|
| Stage A base_only | 78.25 | 61.57 | 0.215 | 3 / 0.05 |
| Stage A base+EUV | 78.19 | 61.51 | 0.213 | 5 / 0.05 |
| Stage B control (no HMI) | 76.37 | 60.40 | 0.267 | 3 / 0.05 |
| Stage B treatment (+HMI) | 76.09 | 60.32 | 0.271 | 6 / 0.02 |

Full per-search candidate tables are in the notebook (`search_base_only`, `search_base_euv`, `search_control`, `search_treatment` — top 5 by mean RMSE printed in cells 8/12, full search results held in-kernel, not persisted to CSV).

Observations:
- Tuning substantially improved Stage A vs. the old fixed-hyperparameter run (RMSE ~82.7/82.0 → ~78.2 for both variants) — the old `XGB_PARAMS` (depth 6, no regularization, no early stopping) was meaningfully suboptimal for this data.
- Stage B's within-window numbers moved only slightly from the post-embargo fixed-hyperparam run (control RMSE 76.29→76.37 i.e. marginally worse; treatment 76.18→76.09 i.e. slightly better) — expected noise from comparing a 15-candidate random sample against hand-tuned values refined over several debugging iterations earlier in the project, combined with the 4-fold→5-fold change altering the CV partition itself. Not a red flag (verified per the plan's sanity check — see plan file in `~/.claude/plans` if it still exists).
- HMI (treatment) continues to show a small within-window edge over control, same direction as before tuning, now slightly larger (R² 0.267→0.271 with HMI vs. without).
- Stage A's EUV-vs-base-only comparison is now essentially a wash post-tuning (mixed signs: EUV improves RMSE/MAE marginally but R² ticks down slightly) — both variants are very close, consistent with the long-standing "EUV is largely redundant with the baseline's own CH features" finding.
- `max_depth=3` (the search space's lower bound) won twice (base_only, control) — worth widening the search space (e.g. try depth 2, or a continuous distribution) in a future follow-up if pursuing this further.
- HSS-event split (Stage A, within-window): tuning shifted things further toward the same pattern already seen in Stage B's HSS split — RMSE during quiet periods improved a lot (81.88→74.17) while RMSE during actual HSS events got worse (84.72→87.45). Combined with Stage B's pre-existing reversal finding, this suggests regularized/tuned/early-stopped models in this pipeline systematically trade HSS-event accuracy for quiet-period accuracy — worth investigating as its own question rather than a tuning artifact specific to one stage.
- Cross-regime holdout numbers are byte-for-byte identical to the pre-tuning run (confirmed) — correctly untouched, as scoped.

**Not yet done**: `presentation/stage_AB_summary.tex`'s within-window CV slide still shows the pre-tuning fixed-hyperparameter numbers; only the cross-regime slides are still accurate as-is. Decide with the user whether to update it once the HSS-event-reversal question above is resolved (it may change the framing, not just the numbers).

**2026-06-19 — HMI feature definition replaced: per-individual-coronal-hole R2/skewness on a 4x3 lat/lon grid, instead of pooled global stats.** The old `global_R2`/`global_energy`/`global_entropy`/`global_variance` features (computed by pooling *every* CH-masked pixel on the disk together) have been replaced in `data/run_hmi_year.py` with per-CH features assigned to a grid, since pooling could cancel out signal from oppositely-imbalanced individual coronal holes.

New feature definition:
- **CH identification**: the SPoCA CH-map FITS is now reprojected onto the HMI pixel grid with `order='nearest-neighbor'` (was the default `'bilinear'`, used only for a `>0.5` binary threshold before) — this preserves SPoCA's own original integer CH-ID labels per pixel with no interpolation blending between adjacent CHs. `ch_id_map = np.rint(np.nan_to_num(ch_id_reprojected, nan=0.0)).astype(np.int64)`; each unique value > 0 is one coronal hole instance.
- For each CH with >= `MIN_CH_PIXELS=10` pixels: `R2 = 2*|0.5 - Φ+/(Φ+ + |Φ-|)|` (same formula as before, now per-CH) and `skew = scipy.stats.skew(ch_pixels)` (Fisher-Pearson, population/biased estimator).
- **Grid**: `LAT_LINES=[-45,0,45]` (4 latitude bands) x `LON_LINES=[-30,30]` (3 longitude bands) = 12 cells, using each CH's pixel-averaged Heliographic-Stonyhurst lat/lon centroid (`np.digitize`) to assign it to one cell.
- **Multiple CHs in the same cell**: the largest CH (by pixel count) wins; its R2/skew become that cell's value. No averaging.
- **Empty cells**: left as NaN — there's no whole-disk fallback anymore (that only made sense for a single global value, not per-cell). If a frame has zero qualifying CHs, all 12 cells are NaN for that row.
- Output columns: `R2_{i}_{j}`, `skew_{i}_{j}` for i=1..4 (lat band), j=1..3 (lon band), plus `n_ch_total`/`n_ch_kept` diagnostic counts (CHs found vs. CHs kept after the size filter).

**Code gotcha hit during implementation**: the `py38` env's sunpy is 2.1.0, which predates `SphericalScreen` (confirmed: `ImportError: cannot import name 'SphericalScreen'`) — this is exactly why the original `hmi_processing.ipynb` grid scaffolding was "never implemented" (Gotcha #1/#3 above). Fix: dropped the `SphericalScreen` context manager entirely; the script already has a blanket `warnings.filterwarnings('ignore')`, so off-disk-pixel transform warnings are silently swallowed and those coordinates just come back NaN, which `np.nanmean` already handles correctly. No functional issue for on-disk coronal holes, which is everything we care about.

**Validated then fully re-run for all 2010-2020.** Small-sample validation (`python run_hmi_year.py 2014 8`) checked first: R2 in [0,1], skew finite, sensible 1-2 CH count for early Jan 2014. Then full parallel re-run, all 11 years at once (script now also supports an optional `sys.argv[2]` row-count cap for quick tests like this). Output goes to new files named `hmi_{YEAR}_6h_grid.csv` (distinct from the old `hmi_{YEAR}_6h.csv` files, which remain on disk untouched as historical/comparison data — both schemas coexist, nothing was deleted).

Final dataset (15,213 rows across all years): R2 correctly bounded in [0,1] everywhere; skew finite everywhere (range roughly -77 to +62, wide due to some CHs sitting right at the 10-pixel minimum where the skewness estimator is noisier); mean ~3.6 kept CHs per frame (vs. ~3.7 found, so the 10px filter is barely trimming anything); only 3.7% of rows have zero usable CHs; ~26% of the 12 grid cells populated per row on average. Per-year row counts and zarr/HTTP-skip counts are close to the original global-feature run (e.g. 2011: 1429 vs. 1433 rows before) — confirms only the per-frame feature computation changed, not the underlying data-fetch logic.

**DONE (see 2026-06-21 update below)**: Stage B now loads the new grid-feature file. The original plan to "scope this explicitly before touching the boosting notebook again" was followed — see below for what changed and how.

## 2026-06-21 update — notebook CV/tuning methodology overhauled twice in one session; HMI switched to grid schema

This update replaces essentially everything in the "Current verified results" and "Final results" sections above with a new methodology. Read this section, not the older numbers above, for the current state of `solar_wind_boosting_pipeline.ipynb`.

**Step 1 — adopted the sibling `transfer/` project's exact CV methodology, to make results comparable.** A `transfer/` folder exists alongside this project (`transfer/cross_validation.py`, `transfer/preprocessing_dataset.py`, `transfer/test.ipynb`) with its own CME-exclusion + CV approach for the same `4x3.csv` dataset. Ported into the boosting notebook:
- `cross_validation_split(Y, n_splits=5)` — 5 equal chronological chunks, each used once as the test fold (train = all *other* chunks, including ones chronologically after the test chunk — not a pure walk-forward split), with a 180-day gap (90 days each side) carved out of every train/test boundary. Replaces the old `TimeSeriesSplit`-based `evaluate_cv`.
- `delete_cmes_from_data_split(data_split, cme_list)` — removes dates inside a CME's start/end window or its 27-day solar-rotation recurrence echo from each fold's train/test dates (verbatim port). This produces a `no_cme` vs `with_cme` scenario pair per evaluation.
- Restricted Stage A and Stage B's within-window CV to dates `<2020-01-01` (`transfer/test.ipynb`'s `eval_mode='cv'`, i.e. solar-cycle-24-only) — verified to reproduce the standalone replication script's `base_only` no_cme RMSE to the decimal (66.1997).
- Cross-regime holdout (`cross_regime_eval`) was explicitly kept untouched throughout — it has no `transfer/`-style analog and its multi-year design conflicts with the `<2020` restriction.

**Step 2 — replaced ALL fixed/borrowed hyperparameters with per-dataset Optuna tuning, no_cme only.** Per explicit user scoping: only the `no_cme` scenario matters going forward (with_cme is no longer computed in the 4 main within-window evaluations); only the 4 main runs get individually tuned (cross-regime stays on the original fixed `TRANSFER_XGB_PARAMS`, untouched). Added to the utility cell:
- `tune_xgb_optuna(X, Y, cme_list, n_trials=60)` — Optuna TPE search (`max_depth`, `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`, `gamma`, `reg_lambda`, `reg_alpha`; `n_estimators=500`/`early_stopping_rounds=50` fixed) scored by mean RMSE across the no_cme folds. **Single-level CV** (same folds select the winner and report its metric) — a known optimistic-bias tradeoff, documented in a markdown cell in the notebook, consistent with this project's prior practice of accepting that tradeoff.
- `feature_pruning_sweep(X, Y, cme_list, shap_ranking, k_fractions=(1.0,0.75,0.5,0.25), n_trials=20)` — for each fraction, keep only the top-K SHAP-ranked columns and retune from scratch, to check whether low-importance features are dead weight.
- `optuna` was not installed in `ch_sws_prediction` — installed via `pip install optuna` (now 4.9.0).
- Required two new SHAP cells (base_only and control previously had none — only base+EUV and treatment did) so all 4 datasets get a feature-pruning sweep.

**Tuning results (no_cme, full 60/20-trial budget, OLD HMI schema — see Step 3 for the schema swap that came after)**: tuning beat the fixed `TRANSFER_XGB_PARAMS` baseline in all 4 cases (base_only 66.20→65.84, base+EUV 65.40→65.10, control 65.31→64.98, treatment 65.25→65.16).

**Feature-pruning finding — a substantial fraction of features are dead weight for 3 of 4 datasets.** Top-25%-by-SHAP-rank beat using all features for base+EUV (65.27→63.76 RMSE), control (65.08→64.01), and treatment (65.27→64.22) — and the trend hadn't plateaued at 25%, so an even smaller subset might do better (not tested, `k_fractions` stopped at 0.25). `base_only` behaved differently: light pruning (75%) helped slightly, but 25% was worse than using everything — it's the smallest/least-redundant feature set of the four, so there was less dead weight to remove. This is reported as a diagnostic only; the pruned subsets were NOT applied back to `X_base_euv`/`X_stageB_treatment`/etc. used elsewhere in the notebook.

**Step 3 — switched Stage B's HMI features to the new grid schema, closing out the "Not yet done" item above.** `data/solar_wind_boosting_pipeline.ipynb`'s data-load cell now reads `data/hmi_2010_2020_6h_grid.csv` (26 columns: `R2_{i}_{j}`/`skew_{i}_{j}` for i=1..4/j=1..3, plus `n_ch_total`/`n_ch_kept`) instead of concatenating the 11 old per-year `hmi_{YEAR}_6h.csv` global-feature files (4 columns: `global_R2`/`global_energy`/`global_entropy`/`global_variance`). Stage B's `treatment` dataset is now 164 features (was 142); `control` is unchanged at 138 (no HMI columns by definition). The `hmi_mask`/regime-detection/SHAP/correlation code needed no structural changes — it was already written generically against "whatever columns `df_hmi` has."

**Validated via reduced-trial smoke test only (6/3 trials instead of 60/20) — zero errors, correct shapes** (HMI loads as `(15213, 26)`; Stage B dataset is still 92,679 rows — `n_ch_total`/`n_ch_kept` being always-populated didn't change row coverage vs. the old schema, since coverage was already determined by file-row presence, not by which stat columns happened to be non-null). HMI feature SHAP ranks under the new schema are scattered (best grid cell rank ~31/164, most others 100+/164) rather than one clearly dominant feature like the old `global_R2` (rank 7-10/142) — worth a closer look once the full-budget run is done. Correlation with EUV CH-area features stays low (|r| ≤ 0.35) for all 26 new columns, same "non-redundant" conclusion as before.

**NOT YET DONE — the full 60/20-trial Optuna run with the new HMI grid schema has not been executed.** Only the reduced-trial (6/3) smoke test has run against the new schema. The "Tuning results" and "Feature-pruning finding" numbers two paragraphs above are from the OLD HMI schema (pre-Step-3) — they're stale for Stage B's `treatment`/`control` specifically (Stage A's `base_only`/`base+EUV` numbers are unaffected by the HMI swap and remain current). **Next step**: re-run `solar_wind_boosting_pipeline.ipynb` end-to-end via nbconvert at the full trial budget (expect ~70-90 minutes based on the prior full run's ~72-minute runtime, likely somewhat longer given 26 HMI columns vs. 4) to get final numbers for the new schema, then update this doc again with those results.

**Session scratch files, not yet cleaned up** (confirm with user before deleting, per this doc's existing convention): `data/solar_wind_boosting_pipeline_smoketest.ipynb` (smoke-test copy, reduced trial counts), `data/transfer_style_cv_base_only.py` (standalone replication script used to validate the `transfer/`-style CV before porting it into the notebook), `data/rewrite_notebook.ps1` (one-off PowerShell script used to bulk-edit notebook cells earlier in the session — its logic now lives in the notebook itself). `data/nbconvert_*.log` files are routine execution logs, low-priority cleanup.

**Encoding gotcha hit and fixed during this session** (add to Gotchas list above): `Get-Content -Raw` in PowerShell without an explicit `-Encoding UTF8` misreads a UTF-8-without-BOM file's non-ASCII characters (em-dashes, etc.) via the system codepage, and writing back with `[System.Text.UTF8Encoding]` then double-encodes them into mojibake. Always pass `-Encoding UTF8` on both read and write when round-tripping `.ipynb` files through PowerShell `Get-Content`/`ConvertFrom-Json`/`ConvertTo-Json`/`[System.IO.File]::WriteAllText`. If corruption is suspected, it's reversible: read the corrupted file as UTF8, re-encode the string via `[System.Text.Encoding]::GetEncoding(1252).GetBytes($raw)`, then decode those bytes as UTF8 again — this undoes exactly one layer of the cp1252-as-UTF8 mojibake round-trip.
