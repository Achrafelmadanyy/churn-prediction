import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from config import THRESHOLD_RANGE, THRESHOLD_STEPS

def calculate_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """Calculate all performance metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_pred_proba)
    }

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    """Find optimal classification threshold based on a metric."""
    thresholds = np.linspace(THRESHOLD_RANGE[0], THRESHOLD_RANGE[1], THRESHOLD_STEPS)
    
    scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            # Balanced metric: harmonic mean of precision and recall
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            score = 2 * (prec * rec) / (prec + rec + 1e-10)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        scores.append(score)
    
    best_idx = np.argmax(scores)
    best_threshold = thresholds[best_idx]
    best_score = scores[best_idx]
    
    return best_threshold, best_score, thresholds, scores

def evaluate_model(model, X, y, threshold=0.5, model_name="Model"):
    """Comprehensive model evaluation."""
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        # For neural networks
        import torch
        if isinstance(model, torch.nn.Module):
            model.eval()
            device = next(model.parameters()).device
            with torch.no_grad():
                y_pred_proba = model(torch.FloatTensor(X).to(device)).cpu().numpy().flatten()
        else:
            y_pred_proba = model.predict(X)
    
    # Calculate metrics with default threshold
    metrics = calculate_all_metrics(y, y_pred_proba, threshold)
    
    # Find optimal threshold for different objectives
    opt_f1_threshold, opt_f1_score, _, _ = find_optimal_threshold(y, y_pred_proba, metric='f1')
    opt_acc_threshold, opt_acc_score, _, _ = find_optimal_threshold(y, y_pred_proba, metric='accuracy')
    opt_bal_threshold, opt_bal_score, _, _ = find_optimal_threshold(y, y_pred_proba, metric='balanced')
    
    metrics['optimal_f1_threshold'] = opt_f1_threshold
    metrics['optimal_f1_score'] = opt_f1_score
    metrics['optimal_accuracy_threshold'] = opt_acc_threshold
    metrics['optimal_accuracy_score'] = opt_acc_score
    metrics['optimal_balanced_threshold'] = opt_bal_threshold
    metrics['optimal_balanced_score'] = opt_bal_score
    
    return metrics, y_pred_proba

def print_detailed_report(y_true, y_pred_proba, threshold, model_name):
    """Print detailed classification report."""
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    print(f"\n{'='*80}")
    print(f"DETAILED CLASSIFICATION REPORT - {model_name}")
    print(f"Threshold: {threshold:.3f}")
    print(f"{'='*80}\n")
    
    print(classification_report(y_true, y_pred, target_names=['No Churn', 'Churn'], digits=4))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  TN = {tn:4d}  |  FP = {fp:4d}")
    print(f"  FN = {fn:4d}  |  TP = {tp:4d}")
    
    print(f"\nAdditional Metrics:")
    print(f"  Specificity: {tn / (tn + fp):.4f}")
    print(f"  NPV:         {tn / (tn + fn):.4f}" if (tn + fn) > 0 else "  NPV: N/A")
    print(f"  FPR:         {fp / (fp + tn):.4f}")
    print(f"  FNR:         {fn / (fn + tp):.4f}")