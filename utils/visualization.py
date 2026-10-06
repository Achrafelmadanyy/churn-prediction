import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, confusion_matrix
from config import FIGURE_DPI, PLOT_STYLE, PLOTS_DIR

sns.set_style(PLOT_STYLE)

def plot_roc_curves(predictions_dict, y_true, save_name='roc_curves.png'):
    """Plot ROC curves for multiple models."""
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(predictions_dict)))
    
    for idx, (name, y_pred_proba) in enumerate(predictions_dict.items()):
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})', 
                linewidth=2.5, color=colors[idx])
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5000)', linewidth=2)
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves Comparison', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=11, loc='lower right', framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    save_path = PLOTS_DIR / save_name
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()

def plot_performance_comparison(results_df, save_name='performance_comparison.png'):
    """Plot performance metrics comparison."""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
    x = np.arange(len(metrics))
    width = 0.11
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(results_df)))
    
    for i, (idx, row) in enumerate(results_df.iterrows()):
        values = [row[m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=row['model'], color=colors[i], edgecolor='black', linewidth=0.7)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    ax.set_xlabel('Metrics', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=14, fontweight='bold')
    ax.set_title('Model Performance Metrics Comparison', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x + width * (len(results_df) - 1) / 2)
    ax.set_xticklabels(metric_labels, fontsize=12)
    ax.legend(fontsize=10, loc='upper left', framealpha=0.9, ncol=2)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.set_ylim([0, 1.15])
    
    save_path = PLOTS_DIR / save_name
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()

def plot_confusion_matrices(predictions_dict, y_true, thresholds=None, save_name='confusion_matrices.png'):
    """Plot confusion matrices for all models."""
    n_models = len(predictions_dict)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5.5 * n_rows))
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, y_pred_proba) in enumerate(predictions_dict.items()):
        threshold = thresholds.get(name, 0.5) if thresholds else 0.5
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                   cbar_kws={'label': 'Count'}, annot_kws={'size': 16, 'fontweight': 'bold'},
                   linewidths=2, linecolor='white')
        axes[idx].set_xlabel('Predicted', fontsize=13, fontweight='bold')
        axes[idx].set_ylabel('Actual', fontsize=13, fontweight='bold')
        axes[idx].set_title(f'{name}\n(threshold = {threshold:.3f})', 
                           fontsize=14, fontweight='bold', pad=10)
        axes[idx].set_xticklabels(['No Churn', 'Churn'], fontsize=11)
        axes[idx].set_yticklabels(['No Churn', 'Churn'], fontsize=11)
    
    # Hide empty subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    save_path = PLOTS_DIR / save_name
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()

def plot_threshold_optimization(thresholds, scores, optimal_threshold, metric='F1', save_name='threshold_optimization.png'):
    """Plot threshold optimization curve."""
    plt.figure(figsize=(12, 7))
    plt.plot(thresholds, scores, linewidth=3, color='steelblue', label=f'{metric} Score')
    plt.axvline(optimal_threshold, color='red', linestyle='--', linewidth=2.5, 
               label=f'Optimal Threshold = {optimal_threshold:.3f}')
    plt.axhline(max(scores), color='green', linestyle=':', linewidth=2, alpha=0.7,
               label=f'Max {metric} = {max(scores):.4f}')
    
    plt.xlabel('Classification Threshold', fontsize=14, fontweight='bold')
    plt.ylabel(f'{metric} Score', fontsize=14, fontweight='bold')
    plt.title(f'{metric} Score vs Classification Threshold', fontsize=16, fontweight='bold', pad=20)
    plt.legend(fontsize=12, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='--')
    
    save_path = PLOTS_DIR / save_name
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()

def plot_feature_importance(feature_names, importances, model_name='Model', top_n=20, save_name='feature_importance.png'):
    """Plot feature importance."""
    indices = np.argsort(importances)[-top_n:]
    
    plt.figure(figsize=(12, 10))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(indices)))
    bars = plt.barh(range(len(indices)), importances[indices], color=colors, edgecolor='black', linewidth=0.7)
    
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices], fontsize=12)
    plt.xlabel('Importance', fontsize=14, fontweight='bold')
    plt.title(f'Top {top_n} Feature Importances - {model_name}', fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height()/2., 
                f'{width:.4f}', ha='left', va='center', fontsize=9, fontweight='bold')
    
    save_path = PLOTS_DIR / save_name
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()

def plot_cv_results(cv_results, model_name='Model', save_name='cv_results.png'):
    """Plot cross-validation results."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Convert to DataFrame
    df = pd.DataFrame(cv_results)
    df = df.sort_values('mean_test_score', ascending=False).head(10)
    
    # Plot 1: Mean test scores
    y_pos = np.arange(len(df))
    ax1.barh(y_pos, df['mean_test_score'], xerr=df['std_test_score'], 
            color='skyblue', edgecolor='black', linewidth=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([f"Config {i+1}" for i in range(len(df))], fontsize=10)
    ax1.set_xlabel('Mean ROC AUC Score', fontsize=12, fontweight='bold')
    ax1.set_title(f'Top 10 Configurations - {model_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Plot 2: Train vs Test scores
    ax2.scatter(df['mean_train_score'], df['mean_test_score'], s=100, alpha=0.6, color='coral', edgecolor='black')
    ax2.plot([df['mean_train_score'].min(), df['mean_train_score'].max()],
            [df['mean_train_score'].min(), df['mean_train_score'].max()],
            'k--', linewidth=2, label='Perfect fit')
    ax2.set_xlabel('Mean Train Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Mean Test Score', fontsize=12, fontweight='bold')
    ax2.set_title('Train vs Test Score', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = PLOTS_DIR / save_name
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.show()