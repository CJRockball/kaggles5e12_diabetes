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

#%%Torch classes and model
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
        #X_out   = torch.concat([num_val, cat_val], axis=1)
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


#%% 


class EarlyStopping:
    def __init__(self, patience=1):
        self.patience = patience
        #print(self.patience)
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None
        
    def __call__(self, val_loss, model):
        score = val_loss
        if self.best_score is None:
            self.best_score = score
            #self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'models/best.pt')
            #print('first best score')
        elif score >= self.best_score:
            self.counter += 1
            #print('counter', self.counter)
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            #self.best_model_state = model.state_dict()
            torch.save(model.state_dict(), 'models/best.pt')
            self.counter = 0
            #print('counter reset')
            
    def load_best_model(self, model):
#        model.load_state_dict(self.best_model_state)           
        model_data = torch.load('models/best.pt', weights_only=False)
        model.load_state_dict(model_data)       

#%% Make model


class Model(nn.Module):
    def __init__(self, meta_data, emb_dropout, fc_in_out, dropout_perc, d_out=1):
        super().__init__()
        n_num_cols = meta_data['num_nums']
        emb_sizes = meta_data['emb_sizes']
        # Get embedding
        self.embedding_d = nn.ModuleList([nn.Embedding(car,siz) for car,siz in emb_sizes])
        for emb in self.embedding_d:
            emb.weight.data.uniform_(-0.01, 0.01)
            #nn.init.kaiming_normal_(emb.weight.data)
            
        # Embedding dropout
        self.emb_dropout = nn.Dropout(emb_dropout)
        # Calculate in_features to linear layer
        emb_vector_sum = sum([e.embedding_dim for e in self.embedding_d])
        # Add in_feature to list
        linear_szs = [emb_vector_sum + n_num_cols] + fc_in_out
        
        # Initialize fc layers
        self.fc_layers = nn.ModuleList([nn.Linear(linear_szs[i],linear_szs[i+1])
                                        for i in range(len(linear_szs) - 1)])
        # Output layer
        self.out = nn.Linear(linear_szs[-1],d_out)
        # Initialize Batch Norm 
        self.batchnorm = nn.ModuleList([nn.BatchNorm1d(s) for s in linear_szs[1:]])
        # Batch for num in
        self.batchnorm_num = nn.BatchNorm1d(n_num_cols)
        # Dropout
        self.dropout = nn.ModuleList([nn.Dropout(p) for p in dropout_perc])
    
    
    def forward(self, num_fields, cat_fields):
        # Initialize embedding for respective cat fields
        x1 = [e(cat_fields[:,i]) for i,e in enumerate(self.embedding_d)]
        # Concatenate all embeddings on axis 1
        x1 = torch.cat(x1,1)
        # Dropout for embeddings
        x1 = self.emb_dropout(x1)
        
        # Input normalization for cont fields
        x2 = self.batchnorm_num(num_fields)
        # Concat inputs
        x1 = torch.cat([x1, x2], 1)
        
        for fc, bn, drop in zip(self.fc_layers, self.batchnorm, self.dropout):
            x1 = F.silu(fc(x1))
            x1 = bn(x1)
            x1 = drop(x1)
        
        x1 = self.out(x1)
        out = F.sigmoid(x1) #sigmoid as we use BCELoss
        return out


#%%


def get_postsplit_meta(Xtrain, meta_data):
    '''Embedding cardinality is a list of two-tuples. First is no of unique values in a cat,
        the second is the number dimensions used to embedd'''
    embedding_cardinality = {n: len(c.unique()) for n,c in Xtrain[meta_data['CATS']].items()}
    emb_sizes = [(size, min(20, (size+1) // 2 )) for item, size in embedding_cardinality.items()]
    meta_data['emb_sizes'] = emb_sizes
    return meta_data


def train(model, loader, optimizer, criterion, DEVICE):
    running_loss = 0.0
    model.train()
    for data in tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        output = model.forward(in1, in2)
        loss = criterion(output, label.float()) #torch.flatten(output)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
    training_loss = running_loss/len(loader)
    return training_loss


def valid(model, loader, criterion, DEVICE):
    y_prediction = []
    y_true = []
    running_loss = 0.0
    model.eval()
    for data in loader: #tqdm(loader):
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
    validation_rocauc = roc_auc_score( y_true1, y_prediction1 )
    return validation_loss, validation_rocauc, y_prediction1 


def test_predictions(model, loader, DEVICE):
    y_prediction = []
    model.eval()
    for data in tqdm(loader):
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        output = model.forward(in1, in2)
        
        y_prediction.append(output.detach().cpu().tolist())
        
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])

    return y_prediction1

#%% Set up data
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


# Split data
df_y = df_train[target].copy()
df_X = df_train.drop(columns=target)
Xtrain, Xvalid, ytrain, yvalid = train_test_split(df_X, df_y, test_size=0.2, random_state=42)
Xtest = df_test.copy()

# Make meta-data
CATS = cats 
NUM = nums #+ nums_te
meta_data = {}
meta_data['NUM'] = NUM
meta_data['CATS'] = CATS
meta_data['num_cats'] = len(CATS)
meta_data['num_nums'] = len(NUM)
# Use category for embedding
# Made sure there are no "new" features in Xtest
meta_data = get_postsplit_meta(df_X, meta_data)


EPOCHS = 100
LR = 1e-4
BATCH_SIZE = 512 # 1024
PATIENCE = 10
DEVICE = torch.device('cuda') # 'cpu') # 

#%%
KFOLD = 5
kf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=1337)

start_time = time.time()
oof = np.zeros(len(df_X))
preds = np.zeros((len(df_test),1))
fold_metric = []
for i, (train_idx, valid_idx) in enumerate(kf.split(df_X, df_y)):
    print(f'#### FOLD {i} ####')
    Xtrain = df_X.loc[train_idx].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx].copy()
    yvalid = df_y.loc[valid_idx].copy()
         
    m = Xtrain[nums].mean()
    s = Xtrain[nums].std()
    Xtrain[nums] = (Xtrain[nums] - m) / s
    Xvalid[nums] = (Xvalid[nums] - m) / s
    Xtest[nums]  = (df_test[nums]  - m) / s
 
    # SET UP DATA standard dataset, dataloader functions
    traindataset = FastDataset(Xtrain, ytrain, meta_data['NUM'], meta_data['CATS'])
    validdataset = FastDataset(Xvalid, yvalid, meta_data['NUM'], meta_data['CATS'])
    trainloader = FastDataLoader(traindataset, batch_size=BATCH_SIZE)
    validloader = FastDataLoader(validdataset, batch_size=BATCH_SIZE)


    # DEF MODEL
    model = Model(meta_data, 0.1, [256, 128, 8], [0.2, 0.2, 0.1]).to(DEVICE)
    # [128, 64, 8] with te
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    
    criterion = nn.BCELoss() # # nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.25])).to(DEVICE) # neg wegiht / pos weight 0.2/0.8 # 
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    early_stopping = EarlyStopping(patience=PATIENCE)


    train_epoch_list = []
    valid_epoch_list = []
    for epoch in range(EPOCHS):
        train_loss = train(model, trainloader, optimizer, criterion, DEVICE)
        validation_loss, validation_rmsle, _ = valid(model, validloader, criterion, DEVICE) #, oof, val_idx)
        
        if epoch % 1 == 0:
            print(f'Epoch: {epoch}/{EPOCHS}, Train loss: {train_loss:.6f}, Validation loss: {validation_loss:.6f}, Validation roc_auc: {validation_rmsle:.6f}')

        train_epoch_list.append(train_loss)
        valid_epoch_list.append(validation_loss)


        early_stopping(validation_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    early_stopping.load_best_model(model)

        
    #plot_data(train_epoch_list, valid_epoch_list)
    validation_loss, validation_rmsle, oof_pred = valid(model, validloader, criterion, DEVICE)
    print(f'RMSLE: {validation_rmsle}')
    fold_metric.append(validation_rmsle)
    oof[valid_idx] = oof_pred.flatten()
    
    ydummy = pd.DataFrame(data=np.zeros((Xtest.shape[0],1)), columns=target) 
    testdataset = FastDataset(Xtest, ydummy, meta_data['NUM'], meta_data['CATS'])
    testloader = FastDataLoader(testdataset, batch_size=BATCH_SIZE)
    y_pred = test_predictions(model, testloader, DEVICE)

    preds += y_pred/KFOLD
    
    # Plot training progress for this fold
    plot_data(
        train_epoch_list, valid_epoch_list,
        title=f'NN Training Progress - Fold {i+1}'
    )
    
end_time = time.time()
print(f'Total time: {end_time - start_time}')
print(fold_metric)
print(f'Average metric: {np.mean(fold_metric)}')

#%%


fname = 'nn2_mlp_base_note'

np.save(f'saved/{fname}_oof.npy', oof.reshape(-1,1))
np.save(f'saved/{fname}_preds.npy', preds)

#%%

print(oof.shape)
print(preds.shape)


#%%

df_sub = pd.read_csv('data/raw/sample_submission.csv')
df_sub['diagnosed_diabetes'] = preds
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)


# %%
