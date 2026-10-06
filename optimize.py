import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR, REPORTS_DIR, PLOTS_DIR
from utils import (load_and_preprocess_data, split_data, prepare_features,
                  evaluate_model, find_optimal_threshold)
from utils.visualization import plot_cv_results, plot_threshold_optimization
from models import optimize_traditional_model, optimize_neural_network

def main():
    print("\n" + "="*80)
    print("CHURN PREDICTION - HYPERPARAMETER OPTIMIZATION")
    print("="*80)
    
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Split data
    df_train, df_val, df_test, df_train_full = split_data(df)
    
    # Prepare features
    print("\nPreparing features...")
    X_train, X_train_scaled, y_train, vectorizer, scaler = prepare_features(df_train, fit_vectorizer=True)
    X_val, X_val_scaled, y_val, _, _ = prepare_features(df_val, fit_vectorizer=False,
                                                         vectorizer=vectorizer, scaler=scaler)
    X_test, X_test_scaled, y_test, _, _ = prepare_features(df_test, fit_vectorizer=False,
                                                           vectorizer=vectorizer, scaler=scaler)
    
    # For optimization, use combined train+val
    X_train_full, X_train_full_scaled, y_train_full, _, _ = prepare_features(
        df_train_full, fit_vectorizer=False, vectorizer=vectorizer, scaler=scaler
    )
    
    print(f"Training samples: {len(X_train_full)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Features: {X_train_full.shape[1]}")
    
    # Dictionary to store optimized models
    optimized_models = {}
    optimization_results = {}
    
    # Optimize traditional models
    model_names = ['LogisticRegression', 'DecisionTree', 'RandomForest', 
                   'GradientBoosting', 'XGBoost', 'LightGBM']
    
    for model_name in model_names:
        print(f"\n{'='*80}")
        print(f"Optimizing {model_name}...")
        print(f"{'='*80}\n")
        
        best_model, cv_results = optimize_traditional_model(
            model_name, X_train_full, y_train_full, n_iter=20, verbose=True
        )
        
        optimized_models[model_name] = best_model
        optimization_results[model_name] = cv_results
        
        # Plot CV results
        if cv_results is not None:
            plot_cv_results(cv_results, model_name, 
                          save_name=f'cv_results_{model_name.lower()}.png')
    
    # Optimize Neural Network
    print(f"\n{'='*80}")
    print("Optimizing Neural Network...")
    print(f"{'='*80}\n")
    
    best_nn_model, best_nn_config, nn_results = optimize_neural_network(
        X_train_scaled, y_train, X_val_scaled, y_val, n_trials=15, verbose=True
    )
    
    optimized_models['Neural Network'] = best_nn_model
    optimization_results['Neural Network'] = nn_results
    
    # Evaluate optimized models on test set
    print("\n" + "="*80)
    print("EVALUATING OPTIMIZED MODELS ON TEST SET")
    print("="*80 + "\n")
    
    results = []
    predictions = {}
    
    for name, model in optimized_models.items():
        print(f"Evaluating {name}...")
        
        if name == 'Neural Network':
            metrics, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, model_name=name)
        else:
            metrics, y_pred_proba = evaluate_model(model, X_test, y_test, model_name=name)
        
        predictions[name] = y_pred_proba
        
        # Find optimal threshold
        opt_threshold, opt_f1, thresholds, scores = find_optimal_threshold(
            y_test, y_pred_proba, metric='f1'
        )
        
        # Plot threshold optimization
        plot_threshold_optimization(
            thresholds, scores, opt_threshold, metric='F1',
            save_name=f'threshold_opt_{name.lower().replace(" ", "_")}.png'
        )
        
        results.append({
            'model': name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'optimal_threshold': opt_threshold,
            'optimal_f1': opt_f1
        })
    
    # Create results DataFrame
    results_df = pd.DataFrame(results).sort_values('roc_auc', ascending=False)
    
    # Print results
    print("\n" + "="*80)
    print("OPTIMIZED MODEL PERFORMANCE COMPARISON")
    print("="*80 + "\n")
    print(results_df.to_string(index=False))
    print("\n" + "="*80)
    
    # Save results
    results_path = REPORTS_DIR / 'optimized_model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Save optimized models
    print("\n" + "="*80)
    print("SAVING OPTIMIZED MODELS")
    print("="*80 + "\n")
    
    for name, model in optimized_models.items():
        model_path = MODELS_DIR / f"{name.lower().replace(' ', '_')}_optimized.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✓ Saved: {model_path}")
    
    # Save best neural network config
    if 'Neural Network' in optimized_models:
        config_path = REPORTS_DIR / 'best_nn_config.txt'
        with open(config_path, 'w') as f:
            f.write("Best Neural Network Configuration:\n")
            f.write("="*50 + "\n")
            for key, value in best_nn_config.items():
                f.write(f"{key}: {value}\n")
        print(f"✓ Saved NN config: {config_path}")
    
    print("\n" + "="*80)
    print("OPTIMIZATION COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()