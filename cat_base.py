#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import shap 

from category_encoders import TargetEncoder

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import auc, log_loss, roc_curve, roc_auc_score, root_mean_squared_error

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool

# Create log directory
LOGDIR = Path(__file__).parent / 'logs'
LOGDIR.mkdir(exist_ok=True)
LOGFILE = LOGDIR / 'ml.log'

# Set up logging, for file and console
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers = [
        logging.FileHandler(LOGFILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"📝 Logging to {LOGFILE}")

RANDOM_STATE = 1337

#%%

def load_data():
    try:
        df_train = pd.read_csv('data/raw/train.csv').drop(columns=['id'])
        df_test = pd.read_csv('data/raw/test.csv').drop(columns=['id'])

    except Exception as e:
        logger.error(f'Failed to load data: {e}')
        raise
    
    return df_train, df_test

df_train, df_test = load_data()

display(df_train)
print(df_train.info())

target = ['diagnosed_diabetes']
cats = df_train.select_dtypes(include=['object']).columns.difference(target).tolist()
print(cats)
print(len(cats))
nums = df_train.select_dtypes(exclude=['object']).columns.difference(target).tolist()
print(nums)
print(len(nums))

#%% Smart groups


cats = [ 'alcohol_consumption_per_week', 'family_history_diabetes', 'hypertension_history',
       'cardiovascular_history','gender', 'ethnicity', 'education_level',
       'income_level', 'smoking_status', 'employment_status']
nums = ['age', 'physical_activity_minutes_per_week', 'diet_score','sleep_hours_per_day',
        'screen_time_hours_per_day', 'bmi', 'waist_to_hip_ratio', 'systolic_bp', 
        'diastolic_bp', 'heart_rate','cholesterol_total','hdl_cholesterol', 'ldl_cholesterol',
       'triglycerides']

#%% Ordinal encoding
def ordinal_encoding(df1, df2):
    train_len = len(df1)
    df = pd.concat([df1, df2], axis=0)
    
    for cat in cats:
        df[cat], _ = df[cat].factorize()     

    df1 = df.iloc[:train_len, :].copy()
    df2 = df.iloc[train_len:, :].copy()
    df2 = df2.drop(columns=target)
    return df1, df2

df_train, df_test = ordinal_encoding(df_train, df_test)
df_train[cats] = df_train[cats].astype('category')
df_test[cats] = df_test[cats].astype('category')
#df_train[nums] = df_train[nums].astype(np.float32)
#df_test[nums] = df_test[nums].astype(np.float32)
print(df_train.shape)
print(df_test.shape)


#%%

def cross_fit_m_estimate_oof(
    df: pd.DataFrame,
    y: np.ndarray,
    col: str,
    n_splits: int = 5,
    m: float = 5.0,
    seed: int = 1337
):
    """
    Cross-fitted M-estimate target encoding for a single column.
    COMPLETE - All dependencies included inline.
    
    Args:
        df: Full DataFrame
        y: Target array (1D numpy array)
        col: Column name to encode (single string)
        n_splits: Number of CV folds
        m: Smoothing parameter
        seed: Random seed
    
    Returns:
        oof: Out-of-fold encoded values
        full_map: {category: (count, positive_sum)} mapping
        prior: Global target mean
    """
    # Validate inputs
    if not isinstance(col, str):
        raise TypeError(f"col must be a string, got {type(col)}")
    
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame")
    
    # Initialize
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    prior = float(np.mean(y))
    oof = np.zeros(len(df), dtype=np.float32)
    full_map = {}
    
    # INLINE HELPER: M-estimate formula
    def compute_m_estimate(count, positive_sum, prior, m):
        """(positive_sum + m*prior) / (count + m)"""
        return (positive_sum + m * prior) / (count + m)
    
    # Cross-fitting loop
    for fold_idx, (train_idx, holdout_idx) in enumerate(skf.split(df, y)):
        # Get fold train data
        fold_train = df.iloc[train_idx]
        fold_y = y[train_idx]
        
        # Compute statistics per category
        tmp_tr = pd.DataFrame({
            'col': fold_train[col].astype(str),
            'y': fold_y
        })
        agg_tr = tmp_tr.groupby('col')['y'].agg(['count', 'sum'])
        
        # Build encoding map for this fold
        tr_map = {}
        for cat_val in agg_tr.index:
            count = int(agg_tr.loc[cat_val, 'count'])
            positive_sum = int(agg_tr.loc[cat_val, 'sum'])
            tr_map[cat_val] = (count, positive_sum)
            full_map[cat_val] = (count, positive_sum)
        
        # Encode holdout fold (leak-safe)
        vals_holdout = df.iloc[holdout_idx][col].astype(str).values
        enc = np.array([
            compute_m_estimate(tr_map[v][0], tr_map[v][1], prior, m)
            if v in tr_map else prior
            for v in vals_holdout
        ], dtype=np.float32)
        
        oof[holdout_idx] = enc
    
    return oof, full_map, prior


def apply_m_estimate_map(
    df: pd.DataFrame,
    col: str,
    full_map: dict,
    prior: float,
    m: float = 5.0
) -> np.ndarray:
    """Apply pre-fitted target encoding to validation/test."""
    
    # INLINE HELPER: Same formula
    def compute_m_estimate(count, positive_sum, prior, m):
        return (positive_sum + m * prior) / (count + m)
    
    vals = df[col].astype(str).values
    out = np.empty(len(vals), dtype=np.float32)
    
    for i, v in enumerate(vals):
        if v in full_map:
            count, positive_sum = full_map[v]
            out[i] = compute_m_estimate(count, positive_sum, prior, m)
        else:
            # Unseen category: fallback to prior
            out[i] = prior
    
    return out



# Storage
te_maps = {}
te_prior = {}
te_train_feats = []

# Encode training data (ONE column at a time)
print("Encoding training data...")
for col in nums:
    print(f"  {col}...", end=" ")
    
    oof, full_map, prior = cross_fit_m_estimate_oof(
        df=df_train,                      # Full DataFrame
        y=df_train[target[0]].values,     # Target as numpy array
        col=col,                          # Single column name
        n_splits=5,
        m=5.0,
        seed=1337
    )
    
    te_maps[col] = full_map
    te_prior[col] = prior
    te_train_feats.append(oof.reshape(-1, 1))
    
    print(f"prior={prior:.4f}, mean={oof.mean():.4f}")

# Combine
# Xtr_te = np.concatenate(te_train_feats, axis=1)
# print(f"\nTraining TE shape: {Xtr_te.shape}")

# # Apply to validation
# print("\nApplying to validation...")
# te_valid_feats = []
# for col in nums:
#     te_valid = apply_m_estimate_map(
#         df=df_valid,
#         col=col,
#         full_map=te_maps[col],
#         prior=te_prior[col],
#         m=5.0
#     )
#     te_valid_feats.append(te_valid.reshape(-1, 1))

# Xva_te = np.concatenate(te_valid_feats, axis=1)
# print(f"Validation TE shape: {Xva_te.shape}")

# Apply to test
te_test_feats = []
for col in nums:
    te_test = apply_m_estimate_map(
        df=df_test,
        col=col,
        full_map=te_maps[col],
        prior=te_prior[col],
        m=5.0
    )
    te_test_feats.append(te_test.reshape(-1, 1))

Xte_te = np.concatenate(te_test_feats, axis=1)
print(f"Test TE shape: {Xte_te.shape}")


# Your column lists
nums_te = []
for cname in nums:
    new_col = f'{cname}_te'
    nums_te.append(new_col)


# Adding new columns to the DataFrame
for i,cname in enumerate(nums_te):
    df_train[cname] = te_train_feats[i]
    df_test[cname] = te_test_feats[i]

display(df_train)
print(df_train.shape, df_test.shape)  

#%%
cat_list = cats #+ nums

cat_params={
    # Core settings
    'iterations': 5000,              # Max iterations
    'learning_rate': 0.03,           # Default is optimal for most cases
    'depth': 6,                      # Default, good balance (range: 4-10)
    'random_seed': 1337,
    'verbose': 200,
    
    # Regularization - KEY for preventing overfitting
    'l2_leaf_reg': 3.0,              # L2 regularization (default, works well)
    'bagging_temperature': 1.0,      # Bayesian bootstrap (default is good)
    'random_strength': 1.0,          # Tree structure randomization
    
    # Sampling
    #'subsample': 0.8,                # Row sampling (like LightGBM)
    'bootstrap_type': 'Bayesian',    # Default, works well for classification
    'max_bin': 1024,                   # Can try to blow out max_bins. Didn't work here
    
    # Objective and metric
    'loss_function': 'Logloss',      # For binary classification
    'eval_metric': 'AUC',            # Same metric as competition
    'use_best_model': True,          # Use best iteration from validation
    
    # Overfitting detector - CRITICAL
    'early_stopping_rounds': 200,    # Stop if no improvement for 200 rounds
    'od_type': 'Iter',               # Overfitting detector type
    #'od_wait': 50,                   # Wait 50 rounds after best iteration
    
    # Speed/quality tradeoff
    'task_type': 'GPU',              # Use 'GPU' if available
    'thread_count': 4,               # Match your CPU cores
}


#%%

df_y = df_train[target].copy()
df_X = df_train.drop(columns=target).copy()

KFOLD = 5
kf = KFold(n_splits=KFOLD, shuffle=True, random_state=RANDOM_STATE )

oof = np.zeros((len(df_train)))
preds = np.zeros((len(df_test)))
fold_metrics = []
fold_loglosses = []
for i,(train_index, valid_index) in enumerate(kf.split(df_X)):
    Xtrain = df_X.iloc[train_index]
    ytrain = df_y.iloc[train_index]
    Xvalid = df_X.iloc[valid_index]
    yvalid = df_y.iloc[valid_index]
    Xtest = df_test.copy()
    
    # enc = TargetEncoder(cols=cats,
    #                     min_samples_leaf=20,
    #                     smoothing=10).fit(Xtrain, ytrain)
    # Xtrain_te = enc.transform(Xtrain)
    # Xvalid_te = enc.transform(Xvalid)
    # Xtest_te = enc.transform(Xtest)
    
    # CAT
    train_pool = Pool(data=Xtrain, label=ytrain, cat_features=cat_list)
    valid_pool = Pool(data=Xvalid, label=yvalid, cat_features=cat_list)
        
    model = CatBoostClassifier(**cat_params,)        
    model.fit(train_pool, eval_set=valid_pool)
    
    

    ypred_proba = model.predict_proba(Xvalid)[:,1]
    fold_logloss = log_loss(yvalid, ypred_proba)
    fold_metric = roc_auc_score(yvalid, ypred_proba)
    oof[valid_index] = ypred_proba

    # Save
    fold_loglosses.append(fold_logloss)
    fold_metrics.append(fold_metric)
    logger.info(f'Fold {i+1}, Log loss: {fold_logloss:.5f}, AUC_ROC: {fold_metric:.5f}')

    preds += model.predict_proba(Xtest)[:,1] / KFOLD
    
logger.info(f"\nOverall Score, logloss: {np.mean(fold_loglosses):.5f}, auc: {np.mean(fold_metrics):.5f}")

#%% Save preds and oof

fname = 'cat_base'

np.save(f'saved/{fname}_oof.npy', oof.reshape(-1,1))
np.save(f'saved/{fname}_preds.npy', preds.reshape(-1,1))

#%%

print(oof.shape)
print(preds.shape)

# %%


df_sub = pd.read_csv('data/raw/sample_submission.csv')
df_sub['diagnosed_diabetes'] = preds.reshape(-1,1)
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)


#%%

