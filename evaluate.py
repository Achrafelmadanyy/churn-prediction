import pandas as pd
import numpy as np
import pickle
import warnings
import argparse
from pathlib import Path
warnings.filterwarnings('ignore')

from config import MODELS_DIR, REPORTS_DIR, PLOTS_DIR
from utils import (load_and_preprocess_data, split_data, prepare_features,
                  evaluate_model, print_detailed_report, find_optimal_threshold)
from utils.visualization import (plot_roc_curves, plot_performance_comparison,
                                 plot_confusion_matrices, plot_threshold_optimization,
                                 plot_feature_importance)

def load_saved_models(optimized=False):
    """Load all saved models from disk."""
    models = {}
    
    suffix = '_optimized' if optimized else '_model'
    
    model_files = list(MODELS_DIR.glob(f'*{suffix}.pkl'))
    
    print(f"\nLoading {'optimized' if optimized else 'trained'} models...")
    for model_path in model_files:
        model_name = model_path.stem.replace(suffix, '').replace('_', ' ').title()
        
        with open(model_path, 'rb') as f:
            models[model_name] = pickle.load(f)
        
        print(f"✓ Loaded: {model_name}")
    
    return models

def load_preprocessing_objects():
    """Load vectorizer and scaler."""
    with open(MODELS_DIR / 'vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(MODELS_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    print("\n✓ Loaded preprocessing objects")
    return vectorizer, scaler

def evaluate_all_models(models, X_test, X_test_scaled, y_test, generate_plots=True):
    """Evaluate all models and generate comprehensive reports."""
    
    print("\n" + "="*80)
    print("EVALUATING ALL MODELS")
    print("="*80 + "\n")
    
    results = []
    predictions = {}
    optimal_thresholds = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # Use scaled data for neural network
        if 'neural' in name.lower() or 'net' in name.lower():
            metrics, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, model_name=name)
        else:
            metrics, y_pred_proba = evaluate_model(model, X_test, y_test, model_name=name)
        
        predictions[name] = y_pred_proba
        optimal_thresholds[name] = metrics['optimal_f1_threshold']
        
        results.append({
            'model': name,
            'accuracy': round(metrics['accuracy'], 4),
            'precision': round(metrics['precision'], 4),
            'recall': round(metrics['recall'], 4),
            'f1': round(metrics['f1'], 4),
            'roc_auc': round(metrics['roc_auc'], 4),
            'optimal_threshold': round(metrics['optimal_f1_threshold'], 4),
            'optimal_f1': round(metrics['optimal_f1_score'], 4)
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    
    # Print results
    print("\n" + "="*80)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*80 + "\n")
    print(results_df.to_string(index=False))
    print("\n" + "="*80)
    
    # Save results
    results_path = REPORTS_DIR / 'evaluation_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Detailed report for best model
    best_model_name = results_df.iloc[0]['model']
    best_threshold = optimal_thresholds[best_model_name]
    print_detailed_report(y_test, predictions[best_model_name], best_threshold, best_model_name)
    
    # Generate visualizations
    if generate_plots:
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS")
        print("="*80 + "\n")
        
        plot_roc_curves(predictions, y_test, save_name='evaluation_roc_curves.png')
        plot_performance_comparison(results_df, save_name='evaluation_performance.png')
        plot_confusion_matrices(predictions, y_test, optimal_thresholds, 
                               save_name='evaluation_confusion_matrices.png')
        
        # Threshold optimization plots for top 3 models
        for idx, row in results_df.head(3).iterrows():
            model_name = row['model']
            y_pred_proba = predictions[model_name]
            
            opt_threshold, opt_score, thresholds, scores = find_optimal_threshold(
                y_test, y_pred_proba, metric='f1'
            )
            
            plot_threshold_optimization(
                thresholds, scores, opt_threshold, metric='F1',
                save_name=f'eval_threshold_{model_name.lower().replace(" ", "_")}.png'
            )
    
    return results_df, predictions, optimal_thresholds

def main():
    parser = argparse.ArgumentParser(description='Evaluate saved churn prediction models')
    parser.add_argument('--optimized', action='store_true', 
                       help='Evaluate optimized models instead of default trained models')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating visualization plots')
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print("CHURN PREDICTION - MODEL EVALUATION")
    print("="*80)
    
    # Load data
    df = load_and_preprocess_data()
    df_train, df_val, df_test, df_train_full = split_data(df)
    
    # Load preprocessing objects
    vectorizer, scaler = load_preprocessing_objects()
    
    # Prepare test features
    print("\nPreparing test features...")
    X_test, X_test_scaled, y_test, _, _ = prepare_features(
        df_test, fit_vectorizer=False, vectorizer=vectorizer, scaler=scaler
    )
    
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_test.shape[1]}")
    
    # Load models
    models = load_saved_models(optimized=args.optimized)
    
    if not models:
        print("\n❌ No models found! Please run train.py or optimize.py first.")
        return
    
    # Evaluate models
    results_df, predictions, thresholds = evaluate_all_models(
        models, X_test, X_test_scaled, y_test, 
        generate_plots=not args.no_plots
    )
    
    # Generate detailed report
    report_path = REPORTS_DIR / 'detailed_evaluation_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CHURN PREDICTION - DETAILED EVALUATION REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("MODEL PERFORMANCE SUMMARY\n")
        f.write("-"*80 + "\n")
        f.write(results_df.to_string(index=False))
        f.write("\n\n")
        
        f.write("="*80 + "\n")
        f.write("DETAILED METRICS FOR EACH MODEL\n")
        f.write("="*80 + "\n\n")
        
        for name, y_pred_proba in predictions.items():
            threshold = thresholds[name]
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            from sklearn.metrics import classification_report, confusion_matrix
            
            f.write(f"\n{name}\n")
            f.write("-"*80 + "\n")
            f.write(f"Optimal Threshold: {threshold:.4f}\n\n")
            f.write(classification_report(y_test, y_pred, 
                                         target_names=['No Churn', 'Churn'], 
                                         digits=4))
            f.write("\n")
            
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            
            f.write(f"Confusion Matrix:\n")
            f.write(f"  TN = {tn:4d}  |  FP = {fp:4d}\n")
            f.write(f"  FN = {fn:4d}  |  TP = {tp:4d}\n\n")
            
            f.write(f"Additional Metrics:\n")
            f.write(f"  Specificity: {tn / (tn + fp):.4f}\n")
            f.write(f"  FPR: {fp / (fp + tn):.4f}\n")
            f.write(f"  FNR: {fn / (fn + tp):.4f}\n")
            f.write("\n" + "="*80 + "\n")
    
    print(f"\n✓ Detailed report saved to: {report_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()