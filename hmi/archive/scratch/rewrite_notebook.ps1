function ToSourceLines($text) {
    $lines = $text -split "`n"
    $arr = @()
    for ($i = 0; $i -lt $lines.Count; $i++) {
        if ($i -lt $lines.Count - 1) { $arr += ($lines[$i] + "`n") } else { $arr += $lines[$i] }
    }
    return $arr
}

$path = "D:\kaijiel3\Downloads\hmi\data\solar_wind_boosting_pipeline.ipynb"
$nb = Get-Content $path -Raw -Encoding UTF8 | ConvertFrom-Json

# ----- cell 1: imports -----
$cell1 = @'
import re
import warnings
from datetime import timedelta
from copy import deepcopy
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

pd.set_option('display.width', 140)
RANDOM_STATE = 42
LAG_DAYS = 4
TOLERANCE = pd.Timedelta(hours=48)
EMBARGO_DAYS = 60  # purge window for cross-regime eval: drop train rows within 2 months of a test region's boundary
rng = np.random.RandomState(RANDOM_STATE)
'@
$nb.cells[1].source = ToSourceLines $cell1

# ----- cell 5: utilities (transfer-style CV engine) -----
$cell5 = @'
def lag_and_merge(feature_df, lag_days, target_index, tolerance=TOLERANCE):
    shifted = feature_df.copy()
    shifted.index = shifted.index + pd.Timedelta(days=lag_days)
    shifted = shifted[~shifted.index.duplicated(keep='last')].sort_index()
    target_frame = pd.DataFrame(index=pd.DatetimeIndex(target_index).sort_values())
    merged = pd.merge_asof(target_frame, shifted, left_index=True, right_index=True,
                            direction='backward', tolerance=tolerance)
    return merged.reindex(target_index)


# Fixed hyperparameters from transfer/test.ipynb -- used everywhere below instead of a tuned
# search, so results are directly comparable to the transfer-folder methodology.
TRANSFER_XGB_PARAMS = dict(
    n_estimators=500, max_depth=3, learning_rate=0.051212530070895795,
    subsample=0.505483952118673, colsample_bytree=0.6326585694513049,
    min_child_weight=4, gamma=2.243192029143412e-07,
    objective='reg:squarederror', tree_method='hist', early_stopping_rounds=50,
)


def cross_validation_split(Y, n_splits=5):
    """Verbatim from transfer/cross_validation.py: 5 equally sized chronological chunks, each
    used once as the test fold (train = all other chunks, including ones *after* the test
    chunk -- not a pure walk-forward split). A symmetric 180-day gap is carved out around every
    train/test boundary (90 days dropped from each side) to prevent near-boundary leakage."""
    dates = Y.index
    discard_len = 180
    discard_len = int(discard_len / 2)
    chunk_len = len(dates) // n_splits
    chunk_remainder = len(dates) % n_splits

    data_split = np.zeros((len(dates), n_splits), dtype=bool)
    for i in range(n_splits):
        data_split[i * chunk_len:(i + 1) * chunk_len, i] = True
    if chunk_remainder > 0:
        data_split[-chunk_remainder:, n_splits - 1] = True

    cv_train, cv_test = [], []
    for i in range(n_splits):
        test_data = np.copy(data_split[:, i])
        train_data = np.any(np.concatenate([data_split[:, :i], data_split[:, i + 1:]], axis=1), axis=1)
        if i == 0:
            test_data[chunk_len - discard_len * 24:chunk_len] = False
            train_data[chunk_len: chunk_len + discard_len * 24] = False
        elif i == (n_splits - 1):
            test_data[chunk_len * i:chunk_len * i + discard_len * 24] = False
            train_data[chunk_len * i - discard_len * 24: chunk_len * i] = False
        else:
            test_data[chunk_len * i: chunk_len * i + discard_len * 24] = False
            test_data[chunk_len * (i + 1) - discard_len * 24:chunk_len * (i + 1)] = False
            train_data[chunk_len * i - discard_len * 24: chunk_len * i] = False
            train_data[chunk_len * (i + 1): chunk_len * (i + 1) + discard_len * 24] = False

        cv_train.append(dates[train_data])
        cv_test.append(dates[test_data])

    return cv_train, cv_test


def delete_cmes_from_data_split(data_split, cme_list):
    """Verbatim from transfer/cross_validation.py: remove dates within a CME's start/end window
    or its 27-day solar-rotation recurrence echo from each fold's train and test dates."""
    train_split = deepcopy(data_split[0])
    test_split = deepcopy(data_split[1])
    n_folds = len(train_split)

    for i in range(n_folds):
        dates_train = train_split[i]
        dates_test = test_split[i]

        binary_cme_train = np.zeros(len(dates_train), dtype=bool)
        binary_cme_test = np.zeros(len(dates_test), dtype=bool)

        for j in range(cme_list.shape[0]):
            binary_cme_train |= ((cme_list.loc[j, 'start'] <= dates_train) & (cme_list.loc[j, 'end'] >= dates_train))
            binary_cme_train |= (((cme_list.loc[j, 'start'] + timedelta(days=26)) <= dates_train) &
                                  ((cme_list.loc[j, 'end'] + timedelta(days=28)) >= dates_train))
            binary_cme_test |= ((cme_list.loc[j, 'start'] <= dates_test) & (cme_list.loc[j, 'end'] >= dates_test))
            binary_cme_test |= (((cme_list.loc[j, 'start'] + timedelta(days=26)) <= dates_test) &
                                 ((cme_list.loc[j, 'end'] + timedelta(days=28)) >= dates_test))

        binary_cme_train = np.invert(binary_cme_train)
        binary_cme_test = np.invert(binary_cme_test)

        train_split[i] = dates_train[binary_cme_train]
        test_split[i] = dates_test[binary_cme_test]

    return [train_split, test_split]


def evaluate_transfer_cv(X, Y, cme_list, n_splits=5, params=None):
    """Transfer-style CV (identical to transfer/test.ipynb): cross_validation_split for folds,
    delete_cmes_from_data_split for the no_cme variant, fixed hyperparameters, early stopping
    against the test fold itself. Returns (per-fold metrics for both scenarios, {scenario: oof
    predictions indexed by date})."""
    params = params or TRANSFER_XGB_PARAMS
    cv_train, cv_test = cross_validation_split(Y, n_splits=n_splits)
    cv_train_no_cme, cv_test_no_cme = delete_cmes_from_data_split([cv_train, cv_test], cme_list)
    scenarios = {'with_cme': (cv_train, cv_test), 'no_cme': (cv_train_no_cme, cv_test_no_cme)}

    rows = []
    oof_pred = {}
    for scenario, (train_dates_list, test_dates_list) in scenarios.items():
        preds_all = []
        for fold, (train_dates, test_dates) in enumerate(zip(train_dates_list, test_dates_list)):
            assert len(train_dates.intersection(test_dates)) == 0, f"Leakage in fold {fold} ({scenario})"
            X_train, Y_train = X.loc[train_dates].copy(), Y.loc[train_dates]
            X_test, Y_test = X.loc[test_dates].copy(), Y.loc[test_dates]
            clean_cols = [str(c).replace('[', '').replace(']', '').replace('<', '').replace('>', '') for c in X_train.columns]
            X_train.columns = clean_cols
            X_test.columns = clean_cols

            model = xgb.XGBRegressor(**params)
            model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)
            pred = pd.Series(model.predict(X_test), index=test_dates)
            preds_all.append(pred)

            cc = np.nan if np.std(pred.values) < 1e-8 else np.corrcoef(Y_test.values, pred.values)[0, 1]
            rows.append({
                'scenario': scenario, 'fold': fold, 'n_train': len(train_dates), 'n_test': len(test_dates),
                'rmse': np.sqrt(mean_squared_error(Y_test, pred)), 'mae': mean_absolute_error(Y_test, pred),
                'r2': r2_score(Y_test, pred), 'cc': cc,
            })
        oof_pred[scenario] = pd.concat(preds_all).sort_index()
    return pd.DataFrame(rows), oof_pred


def oof_to_positional(oof_series, timestamps):
    """Map a date-indexed oof-prediction Series back onto the 0..n-1 positional index aligned
    with `timestamps` (a Series of dates in the same row order as the original positional X/y)."""
    date_to_pos = pd.Series(np.arange(len(timestamps)), index=pd.DatetimeIndex(timestamps.values))
    pos = date_to_pos.loc[oof_series.index].values
    result = pd.Series(np.nan, index=np.arange(len(timestamps)))
    result.iloc[pos] = oof_series.values
    return result


def fit_full_model(X, Y, params=None):
    """Fit on the full dataset (no holdout) for SHAP/diagnostic use -- early_stopping_rounds is
    dropped since there's no eval_set to stop against."""
    params = dict(params or TRANSFER_XGB_PARAMS)
    params.pop('early_stopping_rounds', None)
    model = xgb.XGBRegressor(**params)
    model.fit(X, Y)
    return model
'@
$nb.cells[5].source = ToSourceLines $cell5

# ----- cell 6: markdown retitle -----
$nb.cells[6].source = ToSourceLines "## 2b. CME list (used per-fold below, transfer-style)"

# ----- cell 7: just load cme_list (no global filtering anymore) -----
$cell7 = @'
cme_list = pd.read_csv('enhancements/cme_list.csv', parse_dates=['start', 'end'])
print(f"Loaded {len(cme_list)} CME events ({cme_list['start'].min()} -> {cme_list['end'].max()}); "
      f"used per-fold below via delete_cmes_from_data_split (transfer-style no_cme vs with_cme comparison).")
'@
$nb.cells[7].source = ToSourceLines $cell7

# ----- cell 10: Stage A evaluation (transfer-style, replaces tuning) -----
$cell10 = @'
print("Stage A base-only: transfer-style CV (cross_validation_split, 180-day gap, fixed hyperparameters)...")
cv_base_only, oof_base_only = evaluate_transfer_cv(X_base, y, cme_list)
print(cv_base_only.groupby('scenario')[['rmse', 'mae', 'r2', 'cc']].mean())

print("\nStage A base+EUV: transfer-style CV...")
X_base_euv = pd.concat([X_base, X_euv_lag4], axis=1)
cv_base_euv, oof_base_euv = evaluate_transfer_cv(X_base_euv, y, cme_list)
print(cv_base_euv.groupby('scenario')[['rmse', 'mae', 'r2', 'cc']].mean())

valid = recurrence_27d['speed_recur_27d'].notna()
recur_rmse = np.sqrt(mean_squared_error(y[valid], recurrence_27d.loc[valid, 'speed_recur_27d']))
recur_mae = mean_absolute_error(y[valid], recurrence_27d.loc[valid, 'speed_recur_27d'])
recur_r2 = r2_score(y[valid], recurrence_27d.loc[valid, 'speed_recur_27d'])

summary_stageA = pd.concat([
    cv_base_only.groupby('scenario')[['rmse', 'mae', 'r2', 'cc']].mean().rename(index=lambda s: f'base_only_{s}'),
    cv_base_euv.groupby('scenario')[['rmse', 'mae', 'r2', 'cc']].mean().rename(index=lambda s: f'base_plus_euv_{s}'),
])
summary_stageA.loc['recurrence_27d_naive', ['rmse', 'mae', 'r2']] = [recur_rmse, recur_mae, recur_r2]
print("\nStage A summary (transfer-style CV, fixed hyperparameters, no_cme vs with_cme):")
summary_stageA
'@
$nb.cells[10].source = ToSourceLines $cell10

# ----- cell 13: append date-indexed copies for transfer-style CV -----
$cell13 = (($nb.cells[13].source -join '')).TrimEnd() + @'


X_stageB_control_dt = X_stageB_control.set_index(pd.DatetimeIndex(stageB_timestamps.values))
X_stageB_treatment_dt = X_stageB_treatment.set_index(pd.DatetimeIndex(stageB_timestamps.values))
y_stageB_dt = pd.Series(y_stageB.values, index=X_stageB_control_dt.index)
'@
$nb.cells[13].source = ToSourceLines $cell13

# ----- cell 14: Stage B evaluation (transfer-style, replaces tuning) -----
$cell14 = @'
print("Stage B control (no HMI): transfer-style CV (cross_validation_split, 180-day gap, fixed hyperparameters)...")
cv_control, oof_control = evaluate_transfer_cv(X_stageB_control_dt, y_stageB_dt, cme_list)
print(cv_control.groupby('scenario')[['rmse', 'mae', 'r2', 'cc']].mean())

print("\nStage B treatment (+HMI): transfer-style CV...")
cv_treatment, oof_treatment = evaluate_transfer_cv(X_stageB_treatment_dt, y_stageB_dt, cme_list)
print(cv_treatment.groupby('scenario')[['rmse', 'mae', 'r2', 'cc']].mean())

cv_summary_stageB = pd.concat([
    cv_control.groupby('scenario')[['rmse', 'mae', 'r2', 'cc']].mean().rename(index=lambda s: f'control_{s}'),
    cv_treatment.groupby('scenario')[['rmse', 'mae', 'r2', 'cc']].mean().rename(index=lambda s: f'treatment_{s}'),
])
cv_summary_stageB
'@
$nb.cells[14].source = ToSourceLines $cell14

# ----- cell 15: cross_regime_eval (transfer-style hyperparameters / eval_set) -----
$cell15 = @'
def cross_regime_eval(X, y_target, regime_labels, train_regime, test_regime, timestamps,
                       params=None, embargo_days=EMBARGO_DAYS):
    """Train on one regime, test on another. Regimes are disjoint calendar years, but for
    adjacent years the lag-and-merge tolerance window can let feature data spill across the
    Dec31->Jan1 boundary. Embargo: drop any train row within `embargo_days` of the test
    regime's date range -- a no-op for non-adjacent regime pairs. Fixed transfer-style
    hyperparameters and eval_set=test-fold early stopping, matching the rest of the notebook."""
    params = params or TRANSFER_XGB_PARAMS
    train_mask = regime_labels == train_regime
    test_mask = regime_labels == test_regime
    train_idx_all = np.where(train_mask)[0]
    test_idx = np.where(test_mask)[0]

    if embargo_days:
        test_start = timestamps.iloc[test_idx].min()
        test_end = timestamps.iloc[test_idx].max()
        embargo = pd.Timedelta(days=embargo_days)
        train_ts = timestamps.iloc[train_idx_all]
        keep = (train_ts < test_start - embargo) | (train_ts > test_end + embargo)
        train_idx = train_idx_all[keep.values]
    else:
        train_idx = train_idx_all

    X_train, y_train = X.iloc[train_idx], y_target.iloc[train_idx]
    X_test, y_test = X.iloc[test_idx], y_target.iloc[test_idx]
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    pred = model.predict(X_test)
    full_pred = pd.Series(np.nan, index=np.arange(len(y_target)))
    full_pred.iloc[test_idx] = pred
    cc = np.nan if np.std(pred) < 1e-8 else np.corrcoef(y_test.values, pred)[0, 1]
    return {
        'rmse': np.sqrt(mean_squared_error(y_test, pred)),
        'mae': mean_absolute_error(y_test, pred),
        'r2': r2_score(y_test, pred),
        'cc': cc,
        'n_train': int(len(train_idx)), 'n_test': int(len(test_idx)),
        'n_embargoed': int(len(train_idx_all) - len(train_idx)),
    }, full_pred


MIN_REGIME_ROWS = 1000  # drop degenerate dataset-boundary slivers (e.g. a 139-row stray tail
                         # year) from cross-regime eval -- with the embargo, a regime smaller
                         # than the embargo window can lose ALL its rows as a train set, crashing
regime_counts = pd.Series(regime_stageB).value_counts()
regimes_all = sorted(pd.unique(regime_stageB))
regimes = [r for r in regimes_all if regime_counts[r] >= MIN_REGIME_ROWS]
dropped_regimes = [r for r in regimes_all if r not in regimes]
print(f"Regimes found: {regimes_all} (each a contiguous HMI coverage window, gap-detected at >30 days)")
if dropped_regimes:
    print(f"Excluding from cross-regime eval (< {MIN_REGIME_ROWS} rows, dataset-boundary slivers): "
          f"{[(r, int(regime_counts[r])) for r in dropped_regimes]}")
print(f"Using transfer-style fixed hyperparameters, eval_set=test-fold early stopping, "
      f"{EMBARGO_DAYS}-day train/test embargo for cross-regime holdout.")

cross_results = []
cross_preds = {}
for train_r in regimes:
    for test_r in regimes:
        if train_r == test_r:
            continue
        res_control, pred_control = cross_regime_eval(X_stageB_control, y_stageB, regime_stageB, train_r, test_r,
                                                        timestamps=stageB_timestamps)
        res_treat, pred_treat = cross_regime_eval(X_stageB_treatment, y_stageB, regime_stageB, train_r, test_r,
                                                    timestamps=stageB_timestamps)
        cross_results.append({'train_regime': train_r, 'test_regime': test_r, 'variant': 'control', **res_control})
        cross_results.append({'train_regime': train_r, 'test_regime': test_r, 'variant': 'treatment', **res_treat})
        cross_preds[(train_r, test_r, 'control')] = pred_control
        cross_preds[(train_r, test_r, 'treatment')] = pred_treat

cross_df = pd.DataFrame(cross_results)
print("\nCross-regime holdout (train on one solar-cycle phase, test on the other):")
print(f"Total train rows dropped to embargo across all pairs: {cross_df['n_embargoed'].sum()}")
cross_df
'@
$nb.cells[15].source = ToSourceLines $cell15

# ----- cell 17: SHAP for HMI features (transfer-style fit_full_model) -----
$src17 = ($nb.cells[17].source -join '')
$src17 = $src17.Replace(
    "model_hmi_full = fit_full_model(X_stageB_treatment, y_stageB, params=XGB_PARAMS_REGULARIZED, early_stopping_rounds=30)",
    "model_hmi_full = fit_full_model(X_stageB_treatment, y_stageB)"
)
$nb.cells[17].source = ToSourceLines $src17

# ----- cell 20: HSS event split (adapted to scenario-keyed oof dicts) -----
$cell20 = @'
print("Stage A (base+EUV) split by HSS event:")
stageA_hss = pd.DataFrame(split_metrics_from_oof(oof_base_euv['with_cme'].reindex(y.index), y, hss_flag_full)).T
print(stageA_hss)

print("\nStage B time-series CV, split by HSS event:")
print("control (no HMI):")
stageB_cv_control_hss = pd.DataFrame(split_metrics_from_oof(
    oof_to_positional(oof_control['with_cme'], stageB_timestamps), y_stageB, hss_flag_stageB)).T
print(stageB_cv_control_hss)
print("treatment (with HMI):")
stageB_cv_treatment_hss = pd.DataFrame(split_metrics_from_oof(
    oof_to_positional(oof_treatment['with_cme'], stageB_timestamps), y_stageB, hss_flag_stageB)).T
print(stageB_cv_treatment_hss)

print("\nStage B cross-regime holdout, split by HSS event:")
cross_hss_rows = []
for (train_r, test_r, variant), pred in cross_preds.items():
    res = split_metrics_from_oof(pred, y_stageB, hss_flag_stageB)
    for event_label, metrics in res.items():
        cross_hss_rows.append({'train_regime': train_r, 'test_regime': test_r, 'variant': variant,
                                'event': event_label, **metrics})
cross_hss_df = pd.DataFrame(cross_hss_rows)
cross_hss_df.to_csv('stageB_cross_regime_hss_split_full.csv', index=False)

print("\nAggregate (mean over all year-pairs, weighted equally per pair) by variant x event:")
print(cross_hss_df.groupby(['variant', 'event'])[['rmse', 'mae']].mean())
cross_hss_df
'@
$nb.cells[20].source = ToSourceLines $cell20

# ----- cell 22: final summary -----
$cell22 = @'
print("="*70)
print("STAGE A - full ~10yr EUV-covered period, transfer-style CV, fixed hyperparameters")
print("="*70)
print(summary_stageA)

print("\n" + "="*70)
print("STAGE B - HMI window, 2010-2020+ (11+ calendar-year regimes), transfer-style CV/hyperparameters")
print("="*70)
print("\nTime-series CV (within combined window, transfer-style, fixed hyperparameters):")
print(cv_summary_stageB)
print("\nCross-regime holdout, all year-pairs (incl. tiny edge regimes 2010 partial-year and 2021 stray 139-row tail):")
overall_means = cross_df.groupby('variant')[['rmse', 'mae', 'r2']].mean()
print(overall_means)

full_year_regimes = [r for r in regimes if r not in (2010, 2021)]
cross_df_full_years = cross_df[cross_df['train_regime'].isin(full_year_regimes) & cross_df['test_regime'].isin(full_year_regimes)]
print(f"\nSame, restricted to full-year-only regimes {full_year_regimes} ({len(cross_df_full_years)} of {len(cross_df)} rows):")
full_year_means = cross_df_full_years.groupby('variant')[['rmse', 'mae', 'r2']].mean()
print(full_year_means)

n_treatment_wins = (cross_df.pivot_table(index=['train_regime','test_regime'], columns='variant', values='rmse')
                    .pipe(lambda d: (d['treatment'] < d['control']).sum()))
n_total_pairs = cross_df['train_regime'].nunique() * (cross_df['train_regime'].nunique() - 1)
print(f"\nHMI (treatment) beats control on RMSE in {n_treatment_wins} / {n_total_pairs} cross-regime train/test pairs.")

cross_df.to_csv('stageB_cross_regime_results_full.csv', index=False)
print(f"\nFull cross_df ({len(cross_df)} rows) saved to stageB_cross_regime_results_full.csv for inspection.")

print("\n" + "="*70)
print("HSS-EVENT SPLIT (out-of-sample predictions only, with_cme scenario)")
print("="*70)
print("\nStage A:")
print(stageA_hss)
print("\nStage B cross-regime holdout:")
print(cross_hss_df.pivot_table(index=['train_regime', 'test_regime', 'event'], columns='variant', values=['rmse', 'mae']))
'@
$nb.cells[22].source = ToSourceLines $cell22

# clear stale outputs/execution counts on all modified code cells so it's obvious a re-run is needed
foreach ($i in 1,5,6,7,10,13,14,15,17,20,22) {
    if ($nb.cells[$i].cell_type -eq 'code') {
        $nb.cells[$i].outputs = @()
        $nb.cells[$i].execution_count = $null
    }
}

$json = $nb | ConvertTo-Json -Depth 30
[System.IO.File]::WriteAllText($path, $json, [System.Text.UTF8Encoding]::new($false))
"Rewrite complete. Cell count: $($nb.cells.Count)"
