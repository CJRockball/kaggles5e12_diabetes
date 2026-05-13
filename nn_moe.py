#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, time, random
from tqdm import tqdm

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import gc
import ctypes

def clean_memory():
    """Enhanced memory cleanup for both RAM and VRAM"""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

clean_memory()


def seed_everything(seed=1337):
    """Set seeds for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything()

#%% Load data

def load_data():
    """Load health/diabetes dataset from CSV"""
    try:
        # Update path to your sample.csv location
        df_train = pd.read_csv('data/raw/train.csv')
        df_test = pd.read_csv('data/raw/test.csv')
        
    except Exception as e:
        print(f'Failed to load data: {e}')
        raise

    return df_train, df_test

# Load dataset
df_train, df_test = load_data()

display(df_train.head())
print(df_train.info())
print('number of null values in dataset: ', df_train.isnull().sum().sum())
print(f'Dataset shape: {df_train.shape}')

# Define feature types based on the health dataset
# Numerical features
nums = ['age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week', 
        'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day', 'bmi', 
        'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate', 
        'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides']

# Categorical features
cats = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 
        'employment_status']

# Target variable
target = ['diagnosed_diabetes']

# Binary indicators (already numeric but semantic)
binary_indicators = ['family_history_diabetes', 'hypertension_history', 'cardiovascular_history']

all_cols = nums + cats + binary_indicators + target

# Ordinal encoding for categorical features
def ordinal_encoding(df1, df2):
    """Convert categorical features to numeric ordinals"""
    train_len = len(df1)
    df = pd.concat([df1, df2], axis=0)
    
    for cat in cats+binary_indicators:
        df[cat], _ = df[cat].factorize()     

    df1 = df.iloc[:train_len, :].copy()
    df2 = df.iloc[train_len:, :].copy()
    df2 = df2.drop(columns=target)
    return df1, df2

df_train, df_test = ordinal_encoding(df_train, df_test)

# Set appropriate dtypes
df_train[cats] = df_train[cats].astype('int32')
df_test[cats] = df_test[cats].astype('int32')
df_train[nums + binary_indicators] = df_train[nums + binary_indicators].astype(np.float32)
df_test[nums + binary_indicators] = df_test[nums + binary_indicators].astype(np.float32)
df_train[target] = df_train[target].astype(np.float32)

print(f'\nProcessed dataset shape: {df_train.shape}')
print(f'Numerical features: {len(nums)}')
print(f'Categorical features: {len(cats)}')
print(f'Binary indicators: {len(binary_indicators)}')
print(f'Total input features: {len(nums) + len(cats) + len(binary_indicators)}')

#%% Torch classes and datasets

class FastDataset(Dataset):
    """Fast dataset loader that returns batches of features and targets"""
    def __init__(self, dfX, dfy, num_cols, cat_cols):
        self.cat_features = torch.tensor(dfX.loc[:,cat_cols].values, dtype=torch.long)
        self.num_features = torch.tensor(dfX.loc[:,num_cols].values, dtype=torch.float32)
        self.dfy = torch.tensor(dfy.values, dtype=torch.float32)
         
    def __len__(self):
        return len(self.dfy)
    
    def __getitem__(self, idx, batch_size):
        cat_val = self.cat_features[idx:idx+batch_size,:]
        num_val = self.num_features[idx:idx+batch_size,:]
        y       = self.dfy[idx:idx+batch_size]
        return [num_val, cat_val, y]

class FastDataLoader:
    """Custom data loader for efficient batch processing"""
    def __init__(self, ds, batch_size=32):
        self.ds = ds
        self.dataset_len = ds.__len__()
        self.batch_size = batch_size

        # Calculate number of batches
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
    """Standard PyTorch dataset"""
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
    """Early stopping mechanism to prevent overfitting"""
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
            torch.save(model.state_dict(), 'models/best_moe.pt')
        elif score >= self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            torch.save(model.state_dict(), 'models/best_moe.pt')
            self.counter = 0
            
    def load_best_model(self, model):
        model_data = torch.load('models/best_moe.pt', weights_only=False)
        model.load_state_dict(model_data)       

#%% Mixture of Experts Components

class Expert(nn.Module):
    """
    Expert Network
    
    Each expert is a simple feed-forward neural network.
    In MoE, multiple experts are trained and a gating network
    decides which expert should process each input.
    
    The expert learns a specific subset of the data space.
    This allows specialization rather than forcing one network
    to learn all patterns globally.
    
    Architecture:
    Input -> Hidden Layer 1 -> Hidden Layer 2 -> Output
    """
    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.dropout2 = nn.Dropout(dropout)
        
        self.fc3 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # First hidden layer with batch norm and dropout
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Second hidden layer with batch norm and dropout
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Output layer (no activation - will apply sigmoid in MoE)
        x = self.fc3(x)
        
        return x


class GatingNetwork(nn.Module):
    """
    Gating Network (Router)
    
    The gating network examines the input and produces a probability
    distribution over all experts. This tells us how much weight to
    give each expert's output.
    
    Key insight: The gating network learns to route different types
    of inputs to different experts. This is the "decision maker" that
    allows the mixture of experts to specialize.
    
    The output is a softmax probability distribution over K experts,
    where K is the total number of experts.
    
    Architecture:
    Input -> Dense Layer -> Softmax -> Expert Weights (K values, sum to 1)
    """
    def __init__(self, input_size, num_experts):
        super().__init__()
        
        # Simple gating: project input to expert probabilities
        self.gate = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_experts)
        )
        
    def forward(self, x):
        # Gate outputs logits (not probabilities yet)
        logits = self.gate(x)  # (batch_size, num_experts)
        
        # Convert logits to probabilities using softmax
        # This ensures: sum(probabilities) = 1 for each sample
        probs = F.softmax(logits, dim=1)  # (batch_size, num_experts)
        
        return probs


class LoadBalancingLoss(nn.Module):
    """
    Coefficient of Variation (CV) loss.
    More aggressive than L2 - penalizes any deviation from uniform.
    """
    def __init__(self, num_experts):
        super().__init__()
        self.num_experts = num_experts
        
    def forward(self, gating_probs):
        # Average probability assigned to each expert
        expert_importance = gating_probs.mean(dim=0)  # (num_experts,)
        
        # CV = std / mean
        # If uniform: CV = 0
        # If one expert dominates: CV >> 0
        mean_importance = expert_importance.mean()
        std_importance = expert_importance.std()
        
        cv_loss = std_importance / (mean_importance + 1e-8)
        
        # Also add entropy penalty (encourages high entropy = uniform distribution)
        # H = -sum(p * log(p))
        entropy = -torch.sum(expert_importance * torch.log(expert_importance + 1e-8))
        max_entropy = torch.log(torch.tensor(self.num_experts, dtype=torch.float32))
        entropy_loss = max_entropy - entropy  # Penalize low entropy
        
        return cv_loss + entropy_loss



class MixtureOfExperts(nn.Module):
    """
    Mixture of Experts (MoE) Model
    
    Architecture:
    
    1. Input -> Embedding Layer
       - Categorical features are embedded
       - Numerical features are normalized
       - All concatenated into a single vector
    
    2. Input -> Gating Network
       - Produces probability distribution over experts
       - Each probability indicates expert importance for this input
    
    3. Input -> Multiple Expert Networks (in parallel)
       - Each expert processes the same input independently
       - Each expert specializes on different patterns
       - Outputs are predictions (before sigmoid)
    
    4. Aggregation
       - Final prediction = weighted sum of expert outputs
       - Weights come from the gating network
       - y = sum(gate_prob_i * expert_i(x))
    
    5. Output
       - Apply sigmoid activation for binary classification
    
    Why this works:
    - Each expert can specialize on different data subsets
    - The gating network learns which expert for which input
    - During backprop, experts learn to specialize and gate learns to route
    - Load balancing loss prevents experts from collapsing
    
    Key difference from MLP:
    - MLP: One big network processes all data
    - MoE: Multiple small networks, each specializes, routing is learned
    """
    
    def __init__(self, meta_data, num_experts=4, hidden_size=128, dropout=0.1, d_out=1):
        super().__init__()
        
        # Extract metadata
        n_num_features = meta_data['num_nums']
        n_cat_features = meta_data['num_cats']
        emb_sizes = meta_data['emb_sizes']
        
        # Categorical embeddings
        # Each categorical feature gets an embedding layer
        self.embeddings = nn.ModuleList([
            nn.Embedding(cardinality, emb_dim) 
            for cardinality, emb_dim in emb_sizes
        ])
        
        # Calculate total embedding dimension
        emb_vector_sum = sum([emb_dim for _, emb_dim in emb_sizes])
        
        # Total input dimension = numerical + embedded categorical
        self.input_size = n_num_features + emb_vector_sum
        
        # Store configuration
        self.num_experts = num_experts
        self.num_features = n_num_features
        
        # Batch normalization for numerical features
        self.batchnorm_num = nn.BatchNorm1d(n_num_features)
        
        # Embedding dropout
        self.emb_dropout = nn.Dropout(dropout)
        
        # Gating network: decides which expert to use
        self.gating_network = GatingNetwork(self.input_size, num_experts)
        
        # Expert networks: multiple specialized networks
        self.experts = nn.ModuleList([
            Expert(self.input_size, hidden_size, d_out, dropout)
            for _ in range(num_experts)
        ])
        
        # Load balancing loss component
        self.load_balancing_loss = LoadBalancingLoss(num_experts)
        
    def forward(self, num_features, cat_features, return_gate_probs=False):
        """
        Forward pass through MoE network.
        
        Args:
            num_features: (batch_size, n_num_features) - numerical input
            cat_features: (batch_size, n_cat_features) - categorical input (indices)
            return_gate_probs: if True, also return gating probabilities for analysis
        
        Returns:
            output: (batch_size, 1) - predictions
            gating_probs: (batch_size, num_experts) - routing weights [optional]
        """
        batch_size = num_features.shape[0]
        
        # Step 1: Embed categorical features
        embedded_cats = [emb(cat_features[:, i]) for i, emb in enumerate(self.embeddings)]
        embedded_cats = torch.cat(embedded_cats, dim=1)  # Concatenate all embeddings
        embedded_cats = self.emb_dropout(embedded_cats)
        
        # Step 2: Normalize numerical features
        normalized_nums = self.batchnorm_num(num_features)
        
        # Step 3: Concatenate all features into single input vector
        # This combined input goes to both gating network and experts
        combined_input = torch.cat([normalized_nums, embedded_cats], dim=1)
        
        # Step 4: Get gating probabilities (expert weights for this input)
        gating_probs = self.gating_network(combined_input)  # (batch_size, num_experts)
        
        # Step 5: Process input through all experts in parallel
        expert_outputs = []
        for expert in self.experts:
            expert_output = expert(combined_input)  # (batch_size, 1)
            expert_outputs.append(expert_output)
        
        # Stack expert outputs: (batch_size, num_experts)
        expert_outputs = torch.cat(expert_outputs, dim=1)
        
        # Step 6: Aggregate expert outputs using gating probabilities
        # This is the key: weighted sum of expert predictions
        # Each expert's output is weighted by its gating probability
        aggregated = torch.sum(gating_probs * expert_outputs, dim=1, keepdim=True)
        
        # Step 7: Apply sigmoid for binary classification
        output = torch.sigmoid(aggregated)
        
        if return_gate_probs:
            return output, gating_probs
        else:
            return output
    
    def get_load_balancing_loss(self, gating_probs):
        """
        Calculate load balancing loss to encourage uniform expert usage.
        
        This is called during training to add regularization.
        """
        return self.load_balancing_loss(gating_probs)


#%% Helper functions

def get_postsplit_meta(Xtrain, meta_data):
    """
    Calculate embedding cardinality for categorical features.
    Returns a list of tuples: (num_unique_values, embedding_dim)
    """
    embedding_cardinality = {n: len(c.unique()) for n, c in Xtrain[meta_data['CATS']].items()}
    emb_sizes = [(size, min(50, (size+1) // 2)) for item, size in embedding_cardinality.items()]
    meta_data['emb_sizes'] = emb_sizes
    return meta_data


def train(model, loader, optimizer, criterion, DEVICE, lambda_lb=0.01):
    """
    Training loop for one epoch with MoE-specific loss.
    
    The MoE training includes:
    1. Standard prediction loss (BCE)
    2. Load balancing loss (prevents expert collapse)
    
    Total Loss = BCE Loss + lambda_lb * Load Balancing Loss
    
    Args:
        lambda_lb: weight for load balancing loss
    """
    running_loss = 0.0
    running_lb_loss = 0.0
    model.train()
    
    for data in loader:
        in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with gating probabilities
        output, gating_probs = model.forward(in1, in2, return_gate_probs=True)
        
        # Main prediction loss
        pred_loss = criterion(output, label.float())
        
        # Load balancing loss (encourages uniform expert usage)
        lb_loss = model.get_load_balancing_loss(gating_probs)
        
        # Combined loss
        total_loss = pred_loss + lambda_lb * lb_loss
        
        total_loss.backward()
        optimizer.step()
        
        running_loss += total_loss.item()
        running_lb_loss += lb_loss.item()
    
    training_loss = running_loss / len(loader)
    avg_lb_loss = running_lb_loss / len(loader)
    
    return training_loss, avg_lb_loss


def valid(model, loader, criterion, DEVICE):
    """
    Validation loop.
    
    Returns validation loss and ROC-AUC score.
    """
    y_prediction = []
    y_true = []
    running_loss = 0.0
    model.eval()
    
    with torch.no_grad():
        for data in loader:
            in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
            
            output = model.forward(in1, in2)
            loss = criterion(output, label.float())
            running_loss += loss.item()
            
            y_prediction.append(output.detach().cpu().tolist())
            y_true.append(label.detach().cpu().tolist())
    
    # Flatten predictions and labels    
    y_true1 = np.array([v for lst in y_true for v in lst])
    y_prediction1 = np.array([v for lst in y_prediction for v in lst])
    
    validation_loss = running_loss / len(loader)
    validation_rocauc = roc_auc_score(y_true1, y_prediction1)
    
    return validation_loss, validation_rocauc, y_prediction1 


def test_predictions(model, loader, DEVICE):
    """
    Generate predictions for test set.
    """
    y_prediction = []
    model.eval()
    
    with torch.no_grad():
        for data in tqdm(loader):
            in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
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

Xtrain, Xvalid, ytrain, yvalid = train_test_split(
    df_X, df_y, test_size=0.2, random_state=42
)
Xtest = df_test.copy()

# Make meta-data dictionary
CATS = cats
NUM = nums + binary_indicators  # Include binary indicators with numerical features
meta_data = {}
meta_data['NUM'] = NUM
meta_data['CATS'] = CATS
meta_data['num_cats'] = len(CATS)
meta_data['num_nums'] = len(NUM)

# Calculate embedding sizes based on cardinality
meta_data = get_postsplit_meta(df_X, meta_data)

# Hyperparameters
EPOCHS = 20
LR = 5e-6
BATCH_SIZE = 1024
PATIENCE = 5
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# MoE-specific hyperparameters
NUM_EXPERTS = 4  # Number of expert networks
HIDDEN_SIZE = 128  # Hidden layer size for each expert
LAMBDA_LB = 5  # Weight for load balancing loss Original value 0.001

print(f'Using device: {DEVICE}')
print(f'Number of numerical features: {meta_data["num_nums"]}')
print(f'Number of categorical features: {meta_data["num_cats"]}')
print(f'Total features: {meta_data["num_nums"] + meta_data["num_cats"]}')
print(f'Number of experts: {NUM_EXPERTS}')
print(f'Expert hidden size: {HIDDEN_SIZE}')
print(f'Load balancing lambda: {LAMBDA_LB}')

#%% K-Fold Cross-Validation Training

KFOLD = 5
kf = StratifiedKFold(n_splits=KFOLD, shuffle=True, random_state=1337)

start_time = time.time()
oof = np.zeros(len(df_X))
fold_metric = []

for i, (train_idx, valid_idx) in enumerate(kf.split(df_X, df_y)):
    print(f'\n#### FOLD {i+1}/{KFOLD} ####')
    
    # Split data for this fold
    Xtrain = df_X.loc[train_idx].copy()
    ytrain = df_y.loc[train_idx].copy()
    Xvalid = df_X.loc[valid_idx].copy()
    yvalid = df_y.loc[valid_idx].copy()
    
    # Normalize numerical features
    # Calculate mean and std from training set only (to prevent leakage)
    m = Xtrain[NUM].mean()
    s = Xtrain[NUM].std()
    Xtrain[NUM] = (Xtrain[NUM] - m) / s
    Xvalid[NUM] = (Xvalid[NUM] - m) / s
    Xtest[NUM]  = (Xtest[NUM] - m) / s
    
    # Create datasets and dataloaders
    traindataset = FastDataset(Xtrain, ytrain, meta_data['NUM'], meta_data['CATS'])
    validdataset = FastDataset(Xvalid, yvalid, meta_data['NUM'], meta_data['CATS'])
    trainloader = FastDataLoader(traindataset, batch_size=BATCH_SIZE)
    validloader = FastDataLoader(validdataset, batch_size=BATCH_SIZE)

    # Initialize MoE model
    model = MixtureOfExperts(
        meta_data,
        num_experts=NUM_EXPERTS,
        hidden_size=HIDDEN_SIZE,
        dropout=0.15,
        d_out=1
    ).to(DEVICE)
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Expert structure:')
    for idx, expert in enumerate(model.experts):
        expert_params = sum(p.numel() for p in expert.parameters())
        print(f'  Expert {idx+1}: {expert_params:,} parameters')
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    early_stopping = EarlyStopping(patience=PATIENCE)

    # Training loop
    train_epoch_list = []
    valid_epoch_list = []
    lb_loss_list = []
    
    for epoch in range(EPOCHS):
        train_loss, lb_loss = train(
            model, trainloader, optimizer, criterion, DEVICE, lambda_lb=LAMBDA_LB
        )
        val_loss, val_rocauc, _ = valid(model, validloader, criterion, DEVICE)
        
        # Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        if epoch % 1 == 0:
            print(f'Epoch: {epoch+1:2d}/{EPOCHS} | Train loss: {train_loss:.6f} | '
                  f'LB loss: {lb_loss:.6f} | Val loss: {val_loss:.6f} | '
                  f'Val ROC-AUC: {val_rocauc:.6f}')
        
        train_epoch_list.append(train_loss)
        valid_epoch_list.append(val_loss)
        lb_loss_list.append(lb_loss)

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
    
    # Generate predictions for test set (without labels)
    # Create dummy labels since we need a loader
    ydummy = pd.DataFrame(data=np.zeros(len(Xtest)), columns=target)
    testdataset = FastDataset(Xtest, ydummy, meta_data['NUM'], meta_data['CATS'])
    testloader = FastDataLoader(testdataset, batch_size=BATCH_SIZE)
    
    y_pred = test_predictions(model, testloader, DEVICE)

    # Accumulate predictions across folds
    if i > 0:
        preds += y_pred
    else:
        preds = y_pred.copy()
    
    # Plot training progress for this fold
    plot_data(
        train_epoch_list, valid_epoch_list,
        title=f'MoE Training Progress - Fold {i+1}'
    )
    
    # Clean up memory
    del optimizer, scheduler #model, 
    clean_memory()

# Average predictions across all folds
preds /= KFOLD

end_time = time.time()
print(f'\n=== Training Complete ===')
print(f'Total time: {(end_time - start_time)/60:.2f} minutes')
print(f'Fold ROC-AUC scores: {[f"{m:.6f}" for m in fold_metric]}')
print(f'Average ROC-AUC: {np.mean(fold_metric):.6f} (+/- {np.std(fold_metric):.6f})')

#%% Post-process and save predictions



fname = 'nn1_moe_base3_lowlr'

np.save(f'saved/{fname}_oof.npy', oof.reshape(-1,1))
np.save(f'saved/{fname}_preds.npy', preds)

#%%

print(oof.shape)
print(preds.shape)


#%%

fname = 'nn1_moe_base4_lowlr'

df_sub = pd.read_csv('data/raw/sample_submission.csv')
df_sub['diagnosed_diabetes'] = preds
df_sub.to_csv(f'submissions/{fname}.csv', index=False)

df_check = pd.read_csv(f'submissions/{fname}.csv')
display(df_check)

#%% ANALYSIS: Extract Expert Assignments

def get_expert_assignments(model, loader, DEVICE):
    """
    Run inference and capture which expert was chosen for each sample.
    """
    model.eval()
    all_gate_probs = []
    all_targets = []
    
    with torch.no_grad():
        for data in tqdm(loader, desc="Analyzing Experts"):
            in1, in2, label = data[0].to(DEVICE), data[1].to(DEVICE), data[2].to(DEVICE)
            
            # Get gating probabilities (The "Router")
            _, gate_probs = model.forward(in1, in2, return_gate_probs=True)
            
            all_gate_probs.append(gate_probs.cpu().numpy())
            all_targets.append(label.cpu().numpy())
            
    return np.vstack(all_gate_probs), np.concatenate(all_targets)

# Create a loader for the full dataset (to analyze global patterns)
full_dataset = FastDataset(df_X, df_y, meta_data['NUM'], meta_data['CATS'])
full_loader = FastDataLoader(full_dataset, batch_size=BATCH_SIZE)

# Get probabilities: shape (n_samples, n_experts)
gate_probs, targets = get_expert_assignments(model, full_loader, DEVICE)

# 1. Hard Assignment: Which expert got the highest probability?
assigned_expert = np.argmax(gate_probs, axis=1)

# 2. Confidence: How sure was the gate? (Max probability)
gate_confidence = np.max(gate_probs, axis=1)

print("Expert Assignment Distribution:")
print(pd.Series(assigned_expert).value_counts().sort_index())

#%% ANALYSIS: Characterize Experts

# Create an analysis dataframe combining features and expert assignments
df_analysis = df_X.copy()
df_analysis['Assigned_Expert'] = assigned_expert
df_analysis['Gate_Confidence'] = gate_confidence
df_analysis['Target'] = targets

# Group by Expert and calculate the mean of numerical features
expert_profile = df_analysis.groupby('Assigned_Expert')[nums + binary_indicators].mean()

# Calculate the size of each expert's territory
expert_counts = df_analysis['Assigned_Expert'].value_counts(normalize=True)
expert_profile['Population_Percent'] = expert_counts

# Display the distinct profiles
print("\n=== Mean Feature Values per Expert ===")
# Highlight features that differ significantly between experts
# (Subtract global mean and divide by global std to see deviations)
global_mean = df_analysis[nums + binary_indicators].mean()
global_std = df_analysis[nums + binary_indicators].std()

normalized_profile = (expert_profile[nums + binary_indicators] - global_mean) / global_std

# Show the features with the biggest differences between experts
variance = normalized_profile.var()
top_differentiating_features = variance.sort_values(ascending=False).head(5).index.tolist()

print(f"Top differentiating features: {top_differentiating_features}")
display(expert_profile[top_differentiating_features + ['Population_Percent']].style.background_gradient(cmap='coolwarm'))

#%% VISUALIZATION: Expert Territories (PCA)

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 1. Prepare data for PCA (Numerical only for simplicity)
# Fill NaNs if any (though your training handled them)
X_vis = df_X[nums].fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_vis)

# 2. Run PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 3. Create Plot Data
vis_df = pd.DataFrame({
    'PC1': X_pca[:, 0],
    'PC2': X_pca[:, 1],
    'Expert': assigned_expert,
    'Diabetes': targets.flatten()
})

# 4. Plot
plt.figure(figsize=(12, 5))

# Subplot 1: Expert Regions
plt.subplot(1, 2, 1)
sns.scatterplot(data=vis_df, x='PC1', y='PC2', hue='Expert', palette='tab10', alpha=0.6, s=10)
plt.title('Expert Specialization Regions (PCA)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')

# Subplot 2: True Target Labels (for comparison)
plt.subplot(1, 2, 2)
sns.scatterplot(data=vis_df, x='PC1', y='PC2', hue='Diabetes', palette='viridis', alpha=0.6, s=10)
plt.title('True Labels (Diabetes vs Healthy)')

plt.tight_layout()
plt.show()

#%% VISUALIZATION: Feature Distributions

# Choose top 3 features that separate the experts (calculated in step 2)
features_to_plot = top_differentiating_features[:3] 

plt.figure(figsize=(15, 5))
for i, col in enumerate(features_to_plot):
    plt.subplot(1, 3, i+1)
    sns.violinplot(x='Assigned_Expert', y=col, hue='Assigned_Expert', 
                   data=df_analysis, palette="Set2", legend=False)
    plt.title(f'{col} by Expert')
    plt.xlabel('Expert')
    
plt.tight_layout()
plt.show()


# %%
