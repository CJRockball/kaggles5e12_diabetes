# Kaggle S5E12 — Diabetes Prediction

Raw model collection for the [Kaggle Playground Series Season 5, Episode 12 — Diabetes Prediction](https://www.kaggle.com/competitions/playground-series-s5e12) competition.

📦 **Dataset:** [playground-series-s5e12](https://www.kaggle.com/competitions/playground-series-s5e12/data)

---

## Competition Overview

Binary classification task predicting `diagnosed_diabetes` from a rich set of health indicators. The evaluation metric is **ROC-AUC**. The dataset contains both numerical biomarkers and categorical lifestyle/demographic features.

### Features

**Numerical (14):** `age`, `physical_activity_minutes_per_week`, `diet_score`, `sleep_hours_per_day`, `screen_time_hours_per_day`, `bmi`, `waist_to_hip_ratio`, `systolic_bp`, `diastolic_bp`, `heart_rate`, `cholesterol_total`, `hdl_cholesterol`, `ldl_cholesterol`, `triglycerides`

**Categorical (10):** `gender`, `ethnicity`, `education_level`, `income_level`, `smoking_status`, `employment_status`, `alcohol_consumption_per_week`, `family_history_diabetes`, `hypertension_history`, `cardiovascular_history`

**Target:** `diagnosed_diabetes` (binary)

---

## Repository Structure

```
kaggles5e12_diabetes/
├── data/
│   └── raw/
│       ├── train.csv
│       ├── test.csv
│       └── sample_submission.csv
├── saved/               # OOF and test prediction .npy files
├── submissions/         # CSV submission files
├── models/              # Saved model checkpoints (best.pt, best_moe.pt)
├── logs/                # Training logs
├── xgb_base.py          # XGBoost baseline
├── lgbm_base.py         # LightGBM baseline
├── cat_base.py          # CatBoost baseline
├── nn_base.py           # MLP neural network baseline
├── nn_ftt.py            # FT-Transformer
└── nn_moe.py            # Mixture of Experts
```

---

## Models

### Gradient Boosting Models

All three GBDT models share a common preprocessing pipeline:

- **Ordinal encoding** of categoricals via `pandas.factorize`
- **Cross-fitted M-estimate target encoding** (leak-safe OOF encoding) for numerical columns using 5-fold `StratifiedKFold` with smoothing parameter `m=5.0`
- **5-fold KFold cross-validation** with out-of-fold (OOF) predictions saved to `saved/`

| File | Model | Key Settings |
|------|-------|--------------|
| `xgb_base.py` | XGBoost | 20-fold CV, `max_depth=4`, `reg_alpha/lambda=3.0`, target-encoded cats via `TargetEncoder`, SHAP-compatible |
| `lgbm_base.py` | LightGBM | 5-fold CV, `num_leaves=31`, `learning_rate=0.02`, `max_bin=4096`, `TargetEncoder` for cats |
| `cat_base.py` | CatBoost | 5-fold CV, `depth=6`, `learning_rate=0.03`, native categorical handling via `Pool`, GPU-ready |

### Neural Network Models

All NN models are implemented in **PyTorch** with:
- Global seed (`seed_everything(1337)`) for reproducibility
- `FastDataset` / `FastDataLoader` custom classes for efficient batch processing
- Separate embedding layers for each categorical feature (cardinality-based sizing)
- Per-fold z-score normalization of numerical features (fit on train, applied to val/test)
- `EarlyStopping` with best-model checkpoint saved to `models/`
- 5-fold `StratifiedKFold` cross-validation

#### `nn_base.py` — MLP Baseline

Standard feed-forward network with entity embeddings for categoricals.

- Architecture: `[emb_sum + n_nums] → 256 → 128 → 8 → 1`
- Activation: SiLU per hidden layer, sigmoid output
- Regularization: `BatchNorm1d`, per-layer `Dropout`
- Loss: `BCELoss`, optimizer: Adam (`lr=1e-4`, `weight_decay=1e-5`)
- Input normalization: `BatchNorm1d` on numerical features

#### `nn_ftt.py` — FT-Transformer

Feature Tokenizer + Transformer architecture treating each feature as a token (inspired by BERT/ViT).

- **NumericalFeatureTokenizer**: independent `Linear(1 → d_token)` per numerical feature
- **CategoricalFeatureTokenizer**: per-feature `nn.Embedding` to `d_token`
- CLS token prepended for sequence-level prediction
- `TransformerBlock`: multi-head self-attention + FFN + `LayerNorm` + residual connections
- Config: `d_token=32`, `n_blocks=3`, `n_heads=8`, `d_ffn=256`, `dropout=0.15`
- Loss: `BCELoss`, optimizer: AdamW (`lr=3e-6`), `ReduceLROnPlateau` scheduler

#### `nn_moe.py` — Mixture of Experts

Ensemble of parallel expert networks routed by a learned gating network.

- **Expert**: 2-layer MLP (`input → hidden → hidden → output`) with `BatchNorm1d` + `Dropout`
- **GatingNetwork**: `Linear → ReLU → Dropout → Linear → Softmax` over K experts
- **LoadBalancingLoss**: CV-based penalty + entropy term to prevent expert collapse
- Total loss: `BCE + λ_lb × LoadBalancingLoss` (`λ_lb=5`)
- Config: `num_experts=4`, `hidden_size=128`, `dropout=0.15`, Adam (`lr=5e-6`)
- Includes post-training **expert analysis**: PCA visualization of expert territories and per-expert feature profiling

---

## Shared Preprocessing

The M-estimate target encoding used across all models follows:

$$\hat{y}_c = \frac{n_c \cdot \bar{y}_c + m \cdot \mu}{n_c + m}$$

where $n_c$ is the count of category $c$, $\bar{y}_c$ is the within-category mean, $\mu$ is the global prior, and $m=5$ is the smoothing factor. For training data this is applied in a cross-fitted (OOF) fashion to prevent target leakage.

---

## Usage

### Setup

```bash
pip install pandas numpy scikit-learn xgboost lightgbm catboost torch category_encoders shap tqdm
```

Place the competition data in `data/raw/` and create the output directories:

```bash
mkdir -p data/raw saved submissions models logs
```

### Run a Model

```bash
python xgb_base.py     # XGBoost
python lgbm_base.py    # LightGBM
python cat_base.py     # CatBoost
python nn_base.py      # MLP
python nn_ftt.py       # FT-Transformer
python nn_moe.py       # Mixture of Experts
```

> **Note:** The `.py` files use `display()` (IPython), so they are also compatible with Jupyter notebooks. GPU is used automatically when available for the neural network models.

### Outputs

Each script saves:
- `saved/<model_name>_oof.npy` — out-of-fold predictions on the training set
- `saved/<model_name>_preds.npy` — averaged test predictions
- `submissions/<model_name>.csv` — submission-ready CSV

---

## Requirements

| Package | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data handling |
| `scikit-learn` | CV splits, metrics, encoding |
| `xgboost` | XGBoost model |
| `lightgbm` | LightGBM model |
| `catboost` | CatBoost model |
| `torch` | PyTorch neural networks |
| `category_encoders` | Target encoding (GBDT models) |
| `shap` | Feature importance (imported, ready to use) |
| `tqdm` | Progress bars |
| `matplotlib`, `seaborn` | Visualization (MoE analysis) |
