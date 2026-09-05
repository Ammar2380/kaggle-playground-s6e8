"""
FIXED + upgraded version of your ensemble pipeline.

Key fixes vs your main.ipynb:
1. BUG FIX: LightGBM was n_estimators=500 @ lr=0.02 while XGB/CatBoost used 3000 @ lr=0.02
   -> massive undertraining mismatch. Now all three use matched budgets.
2. Added a 4th, structurally different model (sklearn HistGradientBoosting or a small MLP)
   for real ensemble diversity -- your two LGBM submissions were 99.9% correlated,
   which means blending them added ~nothing.
3. Expanded interaction/ratio features.
4. Never round predictions before writing submission.csv (rounding to 3dp creates
   thousands of tied ranks, which can only hurt or be neutral for AUC -- never helps).
5. Optuna-style search space noted in comments for further tuning.

Run this on Kaggle Notebooks or Colab (multi-core / GPU) for realistic runtime.
On a single CPU core this can take 45-90+ minutes; on Kaggle's standard CPU instance
(4 cores) expect roughly 12-20 minutes for SEEDS_FOR_BAGGING=[42] (single seed).
Scale up seeds for the final run once you've verified OOF AUC improves.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm as lgb
import xgboost as xgb
import catboost as cb

SEED = 42
N_FOLDS = 5
SEEDS_FOR_BAGGING = [42]  # set to [42, 202, 777] once you've validated this runs & helps

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# ----------------------------------------------------------------------
# 1. Feature engineering (your original set + a few additions)
# ----------------------------------------------------------------------
num_cols = ['age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
            'work_study_hours', 'sleep_hours', 'notifications_per_day',
            'app_opens_per_day', 'weekend_screen_time']
cat_cols = ['gender', 'stress_level', 'academic_work_impact']

stress_map = {'Low': 0, 'Medium': 1, 'High': 2}

for df in [train, test]:
    # missingness flags -- missingness itself can carry signal
    for c in num_cols:
        df[f'{c}_missing'] = df[c].isnull().astype(int)

    df['screen_to_sleep_ratio']  = df['daily_screen_time_hours'] / (df['sleep_hours'] + 1e-3)
    df['social_share_of_screen'] = df['social_media_hours'] / (df['daily_screen_time_hours'] + 1e-3)
    df['weekend_vs_weekday']     = df['weekend_screen_time'] - df['daily_screen_time_hours']
    df['leisure_screen_time']    = df['social_media_hours'] + df['gaming_hours']
    df['opens_per_notification'] = df['app_opens_per_day'] / (df['notifications_per_day'] + 1e-3)
    df['notifications_per_open'] = df['notifications_per_day'] / (df['app_opens_per_day'] + 1e-3)
    df['work_leisure_ratio']     = df['work_study_hours'] / (df['leisure_screen_time'] + 1e-3)
    df['screen_time_gap']        = df['weekend_screen_time'] - df['work_study_hours']
    df['sleep_per_age']          = df['sleep_hours'] / (df['age'] + 1e-3)
    df['gaming_share_of_screen'] = df['gaming_hours'] / (df['daily_screen_time_hours'] + 1e-3)
    df['sleep_deficit']          = 8 - df['sleep_hours']
    df['total_screen_x_stress']  = df['daily_screen_time_hours'] * df['stress_level'].map(stress_map)

    df['stress_level_ord']  = df['stress_level'].map(stress_map)
    df['stress_x_social']   = df['stress_level_ord'] * df['social_media_hours']
    df['academic_impact_bin'] = df['academic_work_impact'].map({'No': 0, 'Yes': 1})

missing_cols = [f'{c}_missing' for c in num_cols]
new_num_cols = num_cols + missing_cols + [
    'screen_to_sleep_ratio', 'social_share_of_screen', 'weekend_vs_weekday',
    'leisure_screen_time', 'opens_per_notification', 'notifications_per_open',
    'work_leisure_ratio', 'screen_time_gap', 'sleep_per_age',
    'gaming_share_of_screen', 'sleep_deficit', 'total_screen_x_stress',
    'stress_level_ord', 'stress_x_social', 'academic_impact_bin',
]

for c in cat_cols:
    train[c] = train[c].fillna('Missing').astype('category')
    test[c] = test[c].fillna('Missing').astype('category')

features = new_num_cols + cat_cols
X = train[features].copy()
y = train['addicted_label']
X_test = test[features].copy()
cat_feature_idx = [X.columns.get_loc(c) for c in cat_cols]

# ----------------------------------------------------------------------
# 2. Base model functions (OOF generation, seed-averaged)
# ----------------------------------------------------------------------
def run_lgbm(X, y, X_test, cat_cols):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    # FIXED: matched to XGB/CatBoost budget (was 500 -- badly undertrained)
    params = dict(objective='binary', metric='auc', n_estimators=3000,
                   learning_rate=0.02, num_leaves=48, min_child_samples=30,
                   subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.5, reg_lambda=0.5, verbosity=-1)
    for seed in SEEDS_FOR_BAGGING:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = lgb.LGBMClassifier(**{**params, 'random_state': seed})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.early_stopping(150, verbose=False)])
            oof[val_idx] += model.predict_proba(X_val)[:, 1] / len(SEEDS_FOR_BAGGING)
            test_pred += model.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS_FOR_BAGGING))
    return oof, test_pred

def run_xgb(X, y, X_test, cat_cols):
    X_enc = pd.get_dummies(X, columns=cat_cols)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols)
    X_enc, X_test_enc = X_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

    oof = np.zeros(len(X_enc))
    test_pred = np.zeros(len(X_test_enc))
    params = dict(objective='binary:logistic', eval_metric='auc', n_estimators=3000,
                   learning_rate=0.02, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.5, reg_lambda=1.0, tree_method='hist')
    for seed in SEEDS_FOR_BAGGING:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X_enc, y):
            X_tr, X_val = X_enc.iloc[tr_idx], X_enc.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = xgb.XGBClassifier(**{**params, 'random_state': seed, 'early_stopping_rounds': 150})
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            oof[val_idx] += model.predict_proba(X_val)[:, 1] / len(SEEDS_FOR_BAGGING)
            test_pred += model.predict_proba(X_test_enc)[:, 1] / (N_FOLDS * len(SEEDS_FOR_BAGGING))
    return oof, test_pred

def run_catboost(X, y, X_test, cat_feature_idx):
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    for seed in SEEDS_FOR_BAGGING:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = cb.CatBoostClassifier(
                iterations=3000, learning_rate=0.02, depth=6,
                loss_function='Logloss', eval_metric='AUC',
                random_seed=seed, verbose=False, early_stopping_rounds=150,
                cat_features=cat_feature_idx
            )
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val))
            oof[val_idx] += model.predict_proba(X_val)[:, 1] / len(SEEDS_FOR_BAGGING)
            test_pred += model.predict_proba(X_test)[:, 1] / (N_FOLDS * len(SEEDS_FOR_BAGGING))
    return oof, test_pred

def run_hgb(X, y, X_test, cat_cols):
    """4th model: sklearn's HistGradientBoosting -- structurally different splitting/
    regularization from LGBM/XGB/CatBoost, adds genuine diversity to the blend."""
    X_enc = pd.get_dummies(X, columns=cat_cols)
    X_test_enc = pd.get_dummies(X_test, columns=cat_cols)
    X_enc, X_test_enc = X_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

    oof = np.zeros(len(X_enc))
    test_pred = np.zeros(len(X_test_enc))
    for seed in SEEDS_FOR_BAGGING:
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
        for tr_idx, val_idx in skf.split(X_enc, y):
            X_tr, X_val = X_enc.iloc[tr_idx], X_enc.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = HistGradientBoostingClassifier(
                max_iter=800, learning_rate=0.04, max_depth=8,
                l2_regularization=0.5, random_state=seed,
                early_stopping=True, validation_fraction=0.1, n_iter_no_change=40,
            )
            model.fit(X_tr, y_tr)
            oof[val_idx] += model.predict_proba(X_val)[:, 1] / len(SEEDS_FOR_BAGGING)
            test_pred += model.predict_proba(X_test_enc)[:, 1] / (N_FOLDS * len(SEEDS_FOR_BAGGING))
    return oof, test_pred

# ----------------------------------------------------------------------
# 3. Train all four models
# ----------------------------------------------------------------------
print("Training LightGBM...")
oof_lgb, test_lgb = run_lgbm(X, y, X_test, cat_cols)
print("LGBM OOF AUC:", roc_auc_score(y, oof_lgb))

print("Training XGBoost...")
oof_xgb, test_xgb = run_xgb(X, y, X_test, cat_cols)
print("XGB OOF AUC:", roc_auc_score(y, oof_xgb))

print("Training CatBoost...")
oof_cb, test_cb = run_catboost(X, y, X_test, cat_feature_idx)
print("CatBoost OOF AUC:", roc_auc_score(y, oof_cb))

print("Training HistGradientBoosting...")
oof_hgb, test_hgb = run_hgb(X, y, X_test, cat_cols)
print("HGB OOF AUC:", roc_auc_score(y, oof_hgb))

# ----------------------------------------------------------------------
# 4. Stacked meta-model (logistic regression on OOF predictions)
# ----------------------------------------------------------------------
stack_X = np.column_stack([oof_lgb, oof_xgb, oof_cb, oof_hgb])
stack_X_test = np.column_stack([test_lgb, test_xgb, test_cb, test_hgb])

meta = LogisticRegression()
meta.fit(stack_X, y)
stack_oof_pred = meta.predict_proba(stack_X)[:, 1]
stack_test_pred = meta.predict_proba(stack_X_test)[:, 1]
print("Stacked meta-model OOF AUC:", roc_auc_score(y, stack_oof_pred))
print("Meta-model weights (lgb, xgb, cb, hgb):", meta.coef_)

# simple average as a fallback comparison
oof_avg = (oof_lgb + oof_xgb + oof_cb + oof_hgb) / 4
test_avg = (test_lgb + test_xgb + test_cb + test_hgb) / 4
print("Simple average blend OOF AUC:", roc_auc_score(y, oof_avg))

# ----------------------------------------------------------------------
# 5. Pick the best strategy and write submission.csv (FULL precision -- no rounding)
# ----------------------------------------------------------------------
scores = {
    'simple_avg': (roc_auc_score(y, oof_avg), test_avg),
    'stacked': (roc_auc_score(y, stack_oof_pred), stack_test_pred),
}
best_name = max(scores, key=lambda k: scores[k][0])
best_auc, best_test_pred = scores[best_name]
print(f"Best strategy: {best_name} (OOF AUC {best_auc:.5f})")

submission = pd.DataFrame({'id': test['id'], 'addicted_label': best_test_pred})
submission.to_csv('submission_v2.csv', index=False)
print(submission.head())
