import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import xgboost as xgb
import lightgbm as lgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Configuration
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# === 1. DATA LOADING ===
df = pd.read_csv(Path(__file__).resolve().parent / 'data.csv')
df.columns = df.columns.str.lower().str.replace(' ', '_')

string_columns = df.select_dtypes(include='str').columns
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')

df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')
df.totalcharges = df.totalcharges.fillna(df.totalcharges.mean())
df.churn = (df.churn == 'yes').astype(int)

# === 2. SPLIT DATA ===
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df.churn)
df_train, df_val = train_test_split(df_train_full, test_size=0.25, random_state=42, stratify=df_train_full.churn)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)
df_train_full = df_train_full.reset_index(drop=True)

y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values
y_train_full = df_train_full.churn.values

del df_train['churn'], df_val['churn'], df_test['churn'], df_train_full['churn']

# === 3. FEATURE ENGINEERING ===
num_features = ['seniorcitizen', 'tenure', 'monthlycharges', 'totalcharges']
categorical_features = ['gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
                       'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
                       'techsupport', 'streamingtv', 'streamingmovies', 'contract',
                       'paperlessbilling', 'paymentmethod']

train_dict = df_train[categorical_features + num_features].to_dict(orient='records')
val_dict = df_val[categorical_features + num_features].to_dict(orient='records')
test_dict = df_test[categorical_features + num_features].to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dict)
X_val = dv.transform(val_dict)
X_test = dv.transform(test_dict)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# === 4. PYTORCH NEURAL NETWORK ===
class ChurnNet(nn.Module):
    def __init__(self, input_dim):
        super(ChurnNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc4 = nn.Linear(32, 1)
        
    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = torch.sigmoid(self.fc4(x))
        return x

def train_pytorch_model(X_train, y_train, X_val, y_val, epochs=50, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train).reshape(-1, 1)
    )
    # BatchNorm1d needs more than one sample per batch during training.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    model = ChurnNet(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    best_val_auc = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val).to(device)).cpu().numpy()
            val_auc = roc_auc_score(y_val, val_pred)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss/len(train_loader):.4f} - Val AUC: {val_auc:.4f}")
            
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    return model

# === 5. TRAIN ALL MODELS ===
print("\n=== TRAINING MODELS ===\n")

models = {}
predictions = {}

# Logistic Regression
print("Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
models['Logistic Regression'] = lr_model
predictions['Logistic Regression'] = lr_model.predict_proba(X_test)[:, 1]

# Decision Tree
print("Training Decision Tree...")
dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=42)
dt_model.fit(X_train, y_train)
models['Decision Tree'] = dt_model
predictions['Decision Tree'] = dt_model.predict_proba(X_test)[:, 1]

# Random Forest
print("Training Random Forest...")
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=50, 
                                  random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
models['Random Forest'] = rf_model
predictions['Random Forest'] = rf_model.predict_proba(X_test)[:, 1]

# Gradient Boosting
print("Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
gb_model.fit(X_train, y_train)
models['Gradient Boosting'] = gb_model
predictions['Gradient Boosting'] = gb_model.predict_proba(X_test)[:, 1]

# XGBoost
print("Training XGBoost...")
xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, 
                              eval_metric='logloss')
xgb_model.fit(X_train, y_train)
models['XGBoost'] = xgb_model
predictions['XGBoost'] = xgb_model.predict_proba(X_test)[:, 1]

# LightGBM
print("Training LightGBM...")
lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42, verbose=-1)
lgb_model.fit(X_train, y_train)
models['LightGBM'] = lgb_model
predictions['LightGBM'] = lgb_model.predict_proba(X_test)[:, 1]

# PyTorch Neural Network
print("Training Neural Network...")
nn_model = train_pytorch_model(X_train_scaled, y_train, X_val_scaled, y_val)
nn_model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
with torch.no_grad():
    nn_pred = nn_model(torch.FloatTensor(X_test_scaled).to(device)).cpu().numpy()
models['Neural Network'] = nn_model
predictions['Neural Network'] = nn_pred.flatten()

# === 6. EVALUATE ALL MODELS ===
print("\n=== EVALUATING MODELS ===\n")

results = []
for name, y_pred_proba in predictions.items():
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'ROC AUC': roc_auc
    })

results_df = pd.DataFrame(results).sort_values('ROC AUC', ascending=False)
print("\n" + "="*80)
print("MODEL PERFORMANCE COMPARISON")
print("="*80 + "\n")
print(results_df.to_string(index=False))
print("\n" + "="*80)

# === 7. VISUALIZATIONS ===
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ROC Curves
ax = axes[0, 0]
for name, y_pred_proba in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=2)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

# Performance Metrics Bar Chart
ax = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
x = np.arange(len(metrics))
width = 0.11
for i, (idx, row) in enumerate(results_df.iterrows()):
    values = [row[m] for m in metrics]
    ax.bar(x + i * width, values, width, label=row['Model'])
ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 3)
ax.set_xticklabels(metrics, rotation=45, ha='right')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 1.1])

# Feature Importance
ax = axes[1, 0]
feature_names = dv.get_feature_names_out()
importances = rf_model.feature_importances_
indices = np.argsort(importances)[-15:]
ax.barh(range(len(indices)), importances[indices], color='steelblue')
ax.set_yticks(range(len(indices)))
ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
ax.set_xlabel('Importance', fontsize=12)
ax.set_title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Confusion Matrix
best_model_name = results_df.iloc[0]['Model']
best_pred = (predictions[best_model_name] >= 0.5).astype(int)
cm = confusion_matrix(y_test, best_pred)
ax = axes[1, 1]
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax.set_xticklabels(['No Churn', 'Churn'])
ax.set_yticklabels(['No Churn', 'Churn'])

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# === 8. DETAILED REPORT ===
print(f"\n{'='*80}")
print(f"DETAILED CLASSIFICATION REPORT - {best_model_name}")
print(f"{'='*80}\n")
print(classification_report(y_test, best_pred, target_names=['No Churn', 'Churn']))