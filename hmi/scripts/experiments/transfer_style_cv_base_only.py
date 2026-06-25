"""
Replicates transfer/test.ipynb's CV methodology (cross_validation_split + delete_cmes_from_data_split
+ fixed-hyperparameter XGBoost, RMSE/MAE/R2/CC per fold) for the base-only (4x3.csv) dataset,
restricted to dates before 2020-01-01 (eval_mode='cv' in the original script).
"""
from copy import deepcopy
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def cross_validation_split(Y, n_splits=5):
    """Verbatim from transfer/cross_validation.py."""
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

    cv_train = []
    cv_test = []
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
    """Verbatim from transfer/cross_validation.py."""
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


# ---- replicate transfer/test.ipynb cell-1/cell-2 ----
df = pd.read_csv('4x3.csv', index_col=0, parse_dates=True)
Y = df['speed']
X = df.drop(columns=['speed'])

cme_list = pd.read_csv('enhancements/cme_list.csv', index_col=0, parse_dates=['peak_date', 'start', 'end', 'smoothed_peak_date'])

cv_range = Y.index < datetime(2020, 1, 1)
Y = Y.iloc[cv_range]
X = X.iloc[cv_range, :]

data_split = cross_validation_split(Y)
data_split_without_cmes = delete_cmes_from_data_split(data_split, cme_list)
data_split = [[data_split_without_cmes[0], data_split_without_cmes[1]],
              [data_split[0], data_split[1]]]
data_split = pd.DataFrame(data_split, dtype=object, index=['no_cme', 'with_cme'], columns=['train', 'test'])

print(f"Base-only (4x3.csv) dataset restricted to <2020-01-01: {len(Y)} rows")

# ---- replicate transfer/test.ipynb cell-3 ----
xgb_params = {
    'n_estimators': 500,
    'max_depth': 3,
    'learning_rate': 0.051212530070895795,
    'subsample': 0.505483952118673,
    'colsample_bytree': 0.6326585694513049,
    'min_child_weight': 4,
    'gamma': 2.243192029143412e-07,
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    'early_stopping_rounds': 50,
}

results = {}

for scenario in ['no_cme', 'with_cme']:
    print(f"\n--- Training Scenario: {scenario} ---")

    train_idx_list = data_split.loc[scenario, 'train']
    test_idx_list = data_split.loc[scenario, 'test']

    fold_rmse, fold_mae, fold_r2, fold_cc = [], [], [], []

    for fold_num in range(len(train_idx_list)):
        train_idx = train_idx_list[fold_num]
        test_idx = test_idx_list[fold_num]

        assert len(train_idx.intersection(test_idx)) == 0, f"Leakage detected in fold {fold_num}!"

        X_train, Y_train = X.loc[train_idx].copy(), Y.loc[train_idx]
        X_test, Y_test = X.loc[test_idx].copy(), Y.loc[test_idx]

        X_train.columns = [str(col).replace('[', '').replace(']', '').replace('<', '').replace('>', '')
                            for col in X_train.columns]
        X_test.columns = X_train.columns

        model = xgb.XGBRegressor(**xgb_params)
        model.fit(X_train, Y_train, eval_set=[(X_test, Y_test)], verbose=False)

        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(Y_test, preds))
        mae = mean_absolute_error(Y_test, preds)
        r2 = r2_score(Y_test, preds)

        if np.std(preds) < 1e-8:
            cc = np.nan
        else:
            cc = np.corrcoef(np.asarray(Y_test), np.asarray(preds))[0, 1]

        fold_rmse.append(rmse)
        fold_mae.append(mae)
        fold_r2.append(r2)
        fold_cc.append(cc)

        print(f"Fold {fold_num + 1} - RMSE: {rmse:.2f} km/s | MAE: {mae:.2f} km/s | R2: {r2:.3f} | CC: {cc:.3f} "
              f"| n_train={len(train_idx)} n_test={len(test_idx)}")

    avg_rmse = float(np.nanmean(fold_rmse))
    avg_mae = float(np.nanmean(fold_mae))
    avg_r2 = float(np.nanmean(fold_r2))
    avg_cc = float(np.nanmean(fold_cc))

    print(f">>> {scenario} Average - RMSE: {avg_rmse:.2f} km/s | MAE: {avg_mae:.2f} km/s | "
          f"R2: {avg_r2:.3f} | CC: {avg_cc:.3f}")

    results[scenario] = {
        'rmse_avg': avg_rmse,
        'mae_avg': avg_mae,
        'r2_avg': avg_r2,
        'cc_avg': avg_cc,
    }

print("\n=== Summary ===")
print(pd.DataFrame(results).T)
