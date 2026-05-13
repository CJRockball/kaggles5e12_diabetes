#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import shap 

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os, time, random
from tqdm import tqdm
import gc
import ctypes

def clean_memory():
    """Enhanced memory cleanup for both RAM and VRAM"""
    # Move any remaining tensors to CPU if needed
    # (only if you have model references you want to preserve)
    
    # Synchronize CUDA operations
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Collect garbage
    gc.collect()
    
    # Free GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Trim RAM (Linux-specific)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass  # Silently fail on non-Linux systems
    
    # Optional: Reset peak memory stats for monitoring
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

#del model
clean_memory()


def seed_everything(seed=1337):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # Changed from manual_seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # Added this line
    
seed_everything()

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

#%% Torch classes and model
# Fast Loader is for batches.

class FastDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self,idx, batch_size):
        cat_val = self.cat_features[idx:idx+batch_size,:]
        num_val = self.num_features[idx:idx+batch_size,:]
        y       = self.dfy[idx:idx+batch_size]
        return [num_val, cat_val, y]

class FastDataLoader:
    def __init__(self, ds, batch_size=32):

        self.ds = ds
        self.dataset_len = ds.__len__()
        self.batch_size = batch_size

        # Calculate # batches
        n_batches, remainder = divmod(self.dataset_len, self.batch_size)
        if remainder > 0:
            n_batches += 1
        self.n_batches = n_batches
        
    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.dataset_len:
            raise StopIteration
        batch = self.ds.__getitem__(self.i, self.batch_size)
        self.i += self.batch_size
        return batch

    def __len__(self):
        return self.n_batches

class StdDataset(Dataset):
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self, idx):
        cat = self.cat_features[idx]
        num = self.num_features[idx]
        y = self.dfy[idx]
        return [num, cat, y]


#%% Early Stopping

class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            torch.save(model.state_dict(), 'models/best.pt')
        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), 'models/best.pt')
            self.counter = 0
            
    def load_best_model(self, model):
        model_data = torch.load('models/best.pt', weights_only=False)
        model.load_state_dict(model_data)       

#%% FT-Transformer Model Components

class NumericalFeatureTokenizer(nn.Module):
    """
    Converts numerical features into embedding tokens.
    
    CORRECTED VERSION: Each numerical feature is independently transformed
    into a d_token dimensional embedding. We use ModuleList to create
    separate linear layers for each feature, ensuring proper per-feature
    tokenization without cross-feature mixing.
    
    For each numerical feature:
    1. Apply independent linear transformation (scalar -> d_token vector)
    2. Add a learnable bias term
    
    Input shape: (batch_size, n_features)
    Output shape: (batch_size, n_features, d_token)
    """
    def __init__(self, n_features, d_token):
        super().__init__()
        self.n_features = n_features
        self.d_token = d_token
        
        # Create a separate linear layer for EACH numerical feature
        # This ensures each feature is tokenized independently
        self.linears = nn.ModuleList([
            nn.Linear(1, d_token) for _ in range(n_features)
        ])
        
    def forward(self, x):
        # x shape: (batch_size, n_features)
        batch_size = x.shape
        
        # Tokenize each feature independently
        tokens = []
        for i in range(self.n_features):
            # Extract single feature and add dimension: (batch_size,) -> (batch_size, 1)
            feature = x[:, i:i+1]
            # Apply linear transformation: (batch_size, 1) -> (batch_size, d_token)
            token = self.linears[i](feature)
            tokens.append(token)
        
        # Stack all tokens: list of (batch_size, d_token) -> (batch_size, n_features, d_token)
        x = torch.stack(tokens, dim=1)
        
        return x


class CategoricalFeatureTokenizer(nn.Module):
    """
    Converts categorical features into embedding tokens.
    
    Uses standard embedding layers to map each categorical value
    to a d_token dimensional vector. Each categorical feature gets
    its own embedding table based on its cardinality.
    
    Input shape: (batch_size, n_cat_features)
    Output shape: (batch_size, n_cat_features, d_token)
    """
    def __init__(self, cardinalities, d_token):
        super().__init__()
        # Create an embedding layer for each categorical feature
        # cardinalities is a list of unique value counts for each cat feature
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, d_token) 
            for cardinality in cardinalities
        ])
        # Initialize embeddings with small random values
        for emb in self.embeddings:
            nn.init.uniform_(emb.weight, -0.01, 0.01)
    
    def forward(self, x):
        # x shape: (batch_size, n_cat_features)
        # Process each categorical feature through its embedding
        embedded = [emb(x[:, i]) for i, emb in enumerate(self.embeddings)]
        # Stack all embeddings: (batch_size, n_cat_features, d_token)
        return torch.stack(embedded, dim=1)


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism.
    
    This is the core component that allows features to interact with each other.
    Each "head" learns different patterns of feature relationships, and multiple
    heads allow the model to attend to different aspects simultaneously.
    
    The attention mechanism computes how much each feature should "pay attention"
    to every other feature when making predictions.
    
    Key components:
    - Q (Query): "What am I looking for?"
    - K (Key): "What do I contain?"
    - V (Value): "What information do I provide?"
    
    Attention score = softmax(Q * K^T / sqrt(d_head)) * V
    """
    def __init__(self, d_token, n_heads, dropout=0.0):
        super().__init__()
        assert d_token % n_heads == 0, "d_token must be divisible by n_heads"
        
        self.d_token = d_token
        self.n_heads = n_heads
        self.d_head = d_token // n_heads  # Dimension per head
        
        # Linear projections for queries, keys, and values
        self.W_q = nn.Linear(d_token, d_token)
        self.W_k = nn.Linear(d_token, d_token)
        self.W_v = nn.Linear(d_token, d_token)
        
        # Output projection
        self.W_out = nn.Linear(d_token, d_token)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, n_tokens, d_token)
        batch_size, n_tokens, _ = x.shape
        
        # Project to queries, keys, values
        Q = self.W_q(x)  # (batch_size, n_tokens, d_token)
        K = self.W_k(x)  # (batch_size, n_tokens, d_token)
        V = self.W_v(x)  # (batch_size, n_tokens, d_token)
        
        # Reshape for multi-head attention
        # Split d_token into n_heads * d_head
        Q = Q.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(batch_size, n_tokens, self.n_heads, self.d_head).transpose(1, 2)
        # Now shape: (batch_size, n_heads, n_tokens, d_head)
        
        # Scaled dot-product attention
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_head)
        # scores shape: (batch_size, n_heads, n_tokens, n_tokens)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        # attn_output shape: (batch_size, n_heads, n_tokens, d_head)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, n_tokens, self.d_token)
        
        # Final linear projection
        output = self.W_out(attn_output)
        return output


class TransformerBlock(nn.Module):
    """
    A single transformer block consisting of:
    1. Multi-head self-attention with residual connection
    2. Layer normalization
    3. Feed-forward network (FFN) with residual connection
    4. Layer normalization
    
    This architecture allows gradients to flow easily through deep networks
    and helps the model learn complex feature interactions.
    
    The residual connections (x + layer(x)) allow the model to learn
    incremental changes rather than complete transformations, making
    training more stable.
    """
    def __init__(self, d_token, n_heads, d_ffn, dropout=0.0):
        super().__init__()
        
        # Multi-head attention
        self.attention = MultiHeadAttention(d_token, n_heads, dropout)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        
        # Feed-forward network
        # Typically expands then contracts: d_token -> d_ffn -> d_token
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + attn_output)  # Residual + LayerNorm
        
        # Feed-forward network with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Residual + LayerNorm
        
        return x


class FTTransformer(nn.Module):
    """
    Complete FT-Transformer (Feature Tokenizer + Transformer) model.
    
    Architecture Overview:
    
    1. Feature Tokenization:
       - Numerical features -> NumericalFeatureTokenizer -> tokens
       - Categorical features -> CategoricalFeatureTokenizer -> tokens
       - All tokens have dimension d_token
    
    2. CLS Token:
       - A special learnable token prepended to the sequence
       - Acts as an aggregator that collects information for prediction
       - Borrowed from BERT architecture
    
    3. Transformer Blocks:
       - Stack of transformer layers that process all tokens
       - Learn feature interactions through self-attention
       - Each layer refines the representations
    
    4. Prediction Head:
       - Takes the CLS token output (which has aggregated all info)
       - Passes through MLP to produce final prediction
       - Only the CLS token is used; other tokens are discarded
    
    This design is inspired by BERT and ViT, adapted for tabular data.
    The key insight is treating each feature as a token, allowing
    the transformer to learn which features to attend to.
    """
    def __init__(self, meta_data, d_token=192, n_blocks=3, n_heads=8, 
                 d_ffn=512, dropout=0.1, d_out=1):
        super().__init__()
        
        # Extract metadata
        n_num_features = meta_data['num_nums']
        emb_sizes = meta_data['emb_sizes']  # List of (cardinality, _) tuples
        cardinalities = [card for card, _ in emb_sizes]
        
        # Feature tokenizers
        # Convert raw features into token embeddings
        self.num_tokenizer = NumericalFeatureTokenizer(n_num_features, d_token)
        self.cat_tokenizer = CategoricalFeatureTokenizer(cardinalities, d_token)
        
        # CLS token: learnable parameter that aggregates information
        # This token doesn't correspond to any input feature
        # After transformer blocks, it will contain a summary of all features
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Layer normalization for numerical features (applied before tokenization)
        self.num_norm = nn.LayerNorm(n_num_features)
        
        # Stack of transformer blocks
        # Each block allows features to interact via attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_token, n_heads, d_ffn, dropout)
            for _ in range(n_blocks)
        ])
        
        # Prediction head (MLP applied to CLS token)
        # The CLS token has aggregated all information, so we use it for prediction
        self.head = nn.Sequential(
            nn.LayerNorm(d_token),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_token, d_out)
        )
        
    def forward(self, num_features, cat_features):
        batch_size = int(num_features.shape[0])
        
        # Normalize numerical features
        # This stabilizes training by ensuring features have similar scales
        num_features = self.num_norm(num_features)
        
        # Tokenize features
        # Convert each feature into a d_token dimensional vector
        num_tokens = self.num_tokenizer(num_features)  # (batch_size, n_num, d_token)
        cat_tokens = self.cat_tokenizer(cat_features)  # (batch_size, n_cat, d_token)
        
        # Concatenate all feature tokens
        # Now we have one token per feature
        tokens = torch.cat([num_tokens, cat_tokens], dim=1)  # (batch_size, n_features, d_token)
        
        # Prepend CLS token
        # Expand CLS token for the batch
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch_size, 1, d_token)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch_size, n_features+1, d_token)
        
        # Pass through transformer blocks
        # Each block applies self-attention and FFN
        # Features interact and the CLS token aggregates information
        for block in self.transformer_blocks:
            tokens = block(tokens)
        
        # Extract CLS token (first token)
        # This token has attended to all features and contains aggregated info
        cls_output = tokens[:, 0, :]  # (batch_size, d_token)
        
        # Generate prediction from CLS token
        output = self.head(cls_output)  # (batch_size, d_out)
        
        # Apply sigmoid for binary classification
        output = torch.sigmoid(output)
        
        return output


#%% Helper functions

def get_postsplit_meta(Xtrain, meta_data):
    """
    Calculate embedding cardinality for categorical features.
    Returns a list of tuples: (num_unique_values, embedding_dim)
    """
    embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[meta_data['CATS']].items()}
    emb_sizes = [(size, min(50, (size+1) // 2 )) for item, size in embedding_cardinality.items()]
    meta_data['emb_sizes'] = emb_sizes
    return meta_data


def train(model, loader, optimizer, criterion, DEVICE):
    """
    Training loop for one epoch.
    
    Steps:
    1. Set model to training mode
    2. For each batch:
       - Forward pass
       - Compute loss
       - Backward pass
       - Update weights
    3. Return average training loss
    """
    running_loss = 0.0
    model.train()
    for data in loader:
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        output = model.forward(in1, in2)
        loss = criterion(output, label.float())
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    training_loss = running_loss/len(loader)
    return training_loss


def valid(model, loader, criterion, DEVICE):
    """
    Validation loop.
    
    Steps:
    1. Set model to evaluation mode (disables dropout, etc.)
    2. For each batch:
       - Forward pass (no gradient computation)
       - Compute loss
       - Collect predictions and labels
    3. Calculate validation loss and ROC AUC score
    4. Return metrics and predictions
    """
    y_prediction = []
    y_true = []
    running_loss = 0.0
    model.eval()
    with torch.no_grad():  # No gradient computation during validation
        for data in loader:
            in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
            
            output = model.forward(in1, in2)
            loss = criterion(output, label.float())
            running_loss += loss.item()
            
            y_prediction.append(output.detach().cpu().tolist())
            y_true.append(label.detach().cpu().tolist())
    
    # Flatten prediction and labels    
    y_true1 = np.array([v for lst in y_true for v in lst])
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])
    
    validation_loss = running_loss/len(loader)
    validation_rocauc = roc_auc_score(y_true1, y_prediction1)
    return validation_loss, validation_rocauc, y_prediction1 


def test_predictions(model, loader, DEVICE):
    """
    Generate predictions for test set.
    
    Similar to validation but only returns predictions (no labels available).
    """
    y_prediction = []
    model.eval()
    with torch.no_grad():
        for data in tqdm(loader):
            in1, in2 = data[0].to(DEVICE), data[1].to(DEVICE)
            output = model.forward(in1, in2)
            
            y_prediction.append(output.detach().cpu().tolist())
            
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])

    return y_prediction1


def plot_data(train_d, valid_d, title="Training Progress"):
    """Plot training and validation metrics over epochs."""
    xx = np.arange(len(train_d))
    plt.figure(figsize=(10, 5))
    plt.plot(xx, train_d, label='Train', color='navy', marker='o', markersize=4)
    plt.plot(xx, valid_d, label='Validation', color='darkgreen', marker='s', markersize=4)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    return


#%% Set up data and training

# Split data
df_y = df_train[target].copy()
df_X = df_train.drop(columns=target)
Xtrain, Xvalid, ytrain, yvalid = train_test_split(df_X, df_y, test_size=0.2, random_state=42)

# Make meta-data dictionary
# This contains information about feature types and counts
CATS = cats
NUM = nums + nums_te
meta_data = {}
meta_data['NUM'] = NUM
meta_data['CATS'] = CATS
meta_data['num_cats'] = len(CATS)
meta_data['num_nums'] = len(NUM)
# Calculate embedding sizes based on cardinality
meta_data = get_postsplit_meta(df_X, meta_data)

# Hyperparameters
EPOCHS = 10
LR = 3e-6  # Learning rate for transformer
BATCH_SIZE = 256
PATIENCE = 2
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'Using device: {DEVICE}')
print(f'Number of numerical features: {meta_data["num_nums"]}')
print(f'Number of categorical features: {meta_data["num_cats"]}')
print(f'Total features: {meta_data["num_nums"] + meta_data["num_cats"]}')

#%% K-Fold Cross-Validation Training

KFOLD = 5
kf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=1337)

start_time = time.time()
oof = np.zeros(len(df_X))  # Out-of-fold predictions
fold_metric = []

for i, (train_idx, valid_idx) in enumerate(kf.split(df_X, df_y)):
    print(f'\n#### FOLD {i+1}/{KFOLD} ####')
    
    # Split data for this fold
    Xtrain = df_X.loc[train_idx].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx].copy()
    yvalid = df_y.loc[valid_idx].copy()
    
    # Normalize numerical features
    # Calculate mean and std from training set only
    m = Xtrain[nums].mean()
    s = Xtrain[nums].std()
    Xtrain[nums] = (Xtrain[nums] - m) / s
    Xvalid[nums] = (Xvalid[nums] - m) / s
    df_test[nums] = (df_test[nums] - m) / s
 
    # Create datasets and dataloaders
    traindataset = FastDataset(Xtrain, ytrain, meta_data['NUM'], meta_data['CATS'])
    validdataset = FastDataset(Xvalid, yvalid, meta_data['NUM'], meta_data['CATS'])
    trainloader = FastDataLoader(traindataset, batch_size=BATCH_SIZE)
    validloader = FastDataLoader(validdataset, batch_size=BATCH_SIZE)

    # Initialize FT-Transformer model
    # d_token: embedding dimension for each feature token
    # n_blocks: number of transformer blocks
    # n_heads: number of attention heads
    # d_ffn: dimension of feed-forward network
    model = FTTransformer(
        meta_data, 
        d_token=32,      # Embedding size for each token
        n_blocks=3,       # Number of transformer layers
        n_heads=8,        # Number of attention heads
        d_ffn=256,        # FFN hidden dimension
        dropout=0.15       # Dropout rate
    ).to(DEVICE)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    # Loss function and optimizer
    criterion = nn.BCELoss() #nn.BCEWithLogitsLoss() # nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Learning rate scheduler (optional but recommended for transformers)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Training loop
    train_epoch_list = []
    valid_epoch_list = []
    
    for epoch in range(EPOCHS):
        train_loss = train(model, trainloader, optimizer, criterion, DEVICE)
        val_loss, val_rocauc, _ = valid(model, validloader, criterion, DEVICE)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        if epoch % 1 == 0:
            print(f'Epoch: {epoch+1}/{EPOCHS}, Train loss: {train_loss:.6f}, '
                  f'Val loss: {val_loss:.6f}, Val ROC-AUC: {val_rocauc:.6f}')
        
        train_epoch_list.append(train_loss)
        valid_epoch_list.append(val_loss)

        # Check early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model from this fold
    early_stopping.load_best_model(model)
    
    # Get final validation metrics
    val_loss, val_rocauc, oof_pred = valid(model, validloader, criterion, DEVICE)
    print(f'Fold {i+1} Final ROC-AUC: {val_rocauc:.6f}')
    fold_metric.append(val_rocauc)
    oof[valid_idx] = oof_pred.flatten()
    
    # Generate predictions for test set
    ydummy = pd.DataFrame(data=np.zeros(df_test.shape[0]), columns=target) 
    testdataset = FastDataset(df_test, ydummy, meta_data['NUM'], meta_data['CATS'])
    testloader = FastDataLoader(testdataset, batch_size=BATCH_SIZE)
    y_pred = test_predictions(model, testloader, DEVICE)

    # Accumulate predictions across folds
    if i > 0:
        preds += y_pred
    else:
        preds = y_pred
    
    # Plot training progress for this fold
    plot_data(
        train_epoch_list, valid_epoch_list,
        title=f'NN Training Progress - Fold {i+1}'
    )
    
    # Clean up memory
    del model, optimizer, scheduler
    clean_memory()

# Average predictions across all folds
preds /= KFOLD

end_time = time.time()
print(f'\n=== Training Complete ===')
print(f'Total time: {(end_time - start_time)/60:.2f} minutes')
print(f'Fold ROC-AUC scores: {[f"{m:.6f}" for m in fold_metric]}')
print(f'Average ROC-AUC: {np.mean(fold_metric):.6f} (+/- {np.std(fold_metric):.6f})')

#%%

print(oof.reshape(-1,1).shape)
print(preds.shape)

print(oof)
print(preds)

# %%
fname = 'nn1_ftt_base'

np.save(f'saved/{fname}_oof.npy', oof.reshape(-1,1))
np.save(f'saved/{fname}_preds.npy', preds)


#%%

fname = 'nn1_ftt_basic'

df_sub = pd.read_csv('data/raw/sample_submission.csv')
df_sub['diagnosed_diabetes'] = preds
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)

# %%
