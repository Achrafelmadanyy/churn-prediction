import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import roc_auc_score
from config import NN_CONFIG, NN_HYPERPARAM_SEARCH
import itertools

class ChurnNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=None, dropout_rates=None):
        super(ChurnNet, self).__init__()
        
        if hidden_layers is None:
            hidden_layers = NN_CONFIG['hidden_layers']
        if dropout_rates is None:
            dropout_rates = NN_CONFIG['dropout_rates']
        
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for hidden_dim, dropout_rate in zip(hidden_layers, dropout_rates):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
    def forward(self, x):
        for layer, bn, dropout in zip(self.layers, self.batch_norms, self.dropouts):
            x = layer(x)
            x = bn(x)
            x = torch.relu(x)
            x = dropout(x)
        
        x = torch.sigmoid(self.output_layer(x))
        return x

def train_pytorch_model(X_train, y_train, X_val, y_val, 
                       hidden_layers=None, dropout_rates=None,
                       learning_rate=None, batch_size=None, 
                       epochs=None, patience=None, weight_decay=None,
                       verbose=True):
    """Train a PyTorch neural network for churn prediction."""
    
    # Use config defaults if not provided
    if hidden_layers is None:
        hidden_layers = NN_CONFIG['hidden_layers']
    if dropout_rates is None:
        dropout_rates = NN_CONFIG['dropout_rates']
    if learning_rate is None:
        learning_rate = NN_CONFIG['learning_rate']
    if batch_size is None:
        batch_size = NN_CONFIG['batch_size']
    if epochs is None:
        epochs = NN_CONFIG['epochs']
    if patience is None:
        patience = NN_CONFIG['patience']
    if weight_decay is None:
        weight_decay = NN_CONFIG['weight_decay']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Using device: {device}")
    
    # Prepare data
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    # Initialize model
    model = ChurnNet(X_train.shape[1], hidden_layers, dropout_rates).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Training loop
    best_val_auc = 0
    patience_counter = 0
    history = {'train_loss': [], 'val_auc': []}
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            batch_count += 1
        
        avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_pred)
        
        history['train_loss'].append(avg_train_loss)
        history['val_auc'].append(val_auc)
        
        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_train_loss:.4f} - Val AUC: {val_auc:.4f}")
        
        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= patience:
            if verbose:
                print(f"Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    if verbose:
        print(f"Best validation AUC: {best_val_auc:.4f}")
    
    return model, best_val_auc, history

def optimize_neural_network(X_train, y_train, X_val, y_val, n_trials=10, verbose=True):
    """Optimize neural network hyperparameters using grid search."""
    
    print("\n" + "="*80)
    print("NEURAL NETWORK HYPERPARAMETER OPTIMIZATION")
    print("="*80 + "\n")
    
    best_auc = 0
    best_config = None
    best_model = None
    results = []
    
    # Create hyperparameter combinations
    architectures = NN_HYPERPARAM_SEARCH['architectures']
    learning_rates = NN_HYPERPARAM_SEARCH['learning_rates']
    batch_sizes = NN_HYPERPARAM_SEARCH['batch_sizes']
    weight_decays = NN_HYPERPARAM_SEARCH['weight_decays']
    
    # Sample random combinations
    all_combinations = list(itertools.product(architectures, learning_rates, batch_sizes, weight_decays))
    
    if len(all_combinations) > n_trials:
        import random
        random.seed(42)
        sampled_combinations = random.sample(all_combinations, n_trials)
    else:
        sampled_combinations = all_combinations
    
    print(f"Testing {len(sampled_combinations)} configurations...\n")
    
    for idx, (arch, lr, bs, wd) in enumerate(sampled_combinations, 1):
        hidden_layers = arch['layers']
        dropout_rates = arch['dropouts']
        
        config = {
            'hidden_layers': hidden_layers,
            'dropout_rates': dropout_rates,
            'learning_rate': lr,
            'batch_size': bs,
            'weight_decay': wd
        }
        
        print(f"[{idx}/{len(sampled_combinations)}] Testing: layers={hidden_layers}, lr={lr}, bs={bs}, wd={wd}")
        
        try:
            model, val_auc, history = train_pytorch_model(
                X_train, y_train, X_val, y_val,
                hidden_layers=hidden_layers,
                dropout_rates=dropout_rates,
                learning_rate=lr,
                batch_size=bs,
                weight_decay=wd,
                verbose=False
            )
            
            results.append({
                'config': config,
                'val_auc': val_auc,
                'history': history
            })
            
            print(f"  → Val AUC: {val_auc:.4f}")
            
            if val_auc > best_auc:
                best_auc = val_auc
                best_config = config
                best_model = model
                print(f"  ✓ New best model!")
            
        except Exception as e:
            print(f"  ✗ Failed: {str(e)}")
        
        print()
    
    print("="*80)
    print(f"BEST CONFIGURATION - Val AUC: {best_auc:.4f}")
    print("="*80)
    print(f"Hidden Layers: {best_config['hidden_layers']}")
    print(f"Dropout Rates: {best_config['dropout_rates']}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Batch Size: {best_config['batch_size']}")
    print(f"Weight Decay: {best_config['weight_decay']}")
    print("="*80 + "\n")
    
    return best_model, best_config, results