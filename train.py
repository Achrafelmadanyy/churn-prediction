import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from config import MODELS_DIR, REPORTS_DIR
from utils import (load_and_preprocess_data, split_data, prepare_features,
                  evaluate_model, print_detailed_report)
from utils.visualization import (plot_roc_curves, plot_performance_comparison,
                                 plot_confusion_matrices, plot_feature_importance)
from models import train_all_traditional_models, train_pytorch_model

def main():
    print("\n" + "="*80)
    print("CHURN PREDICTION - TRAINING PIPELINE")
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
    
    print(f"Feature dimensions: {X_train.shape[1]}")
    
    # Train traditional models
    models = train_all_traditional_models(X_train, y_train)
    
    # Train neural network
    print("Training Neural Network...")
    nn_model, best_val_auc, history = train_pytorch_model(X_train_scaled, y_train, 
                                                          X_val_scaled, y_val)
    models['Neural Network'] = nn_model
    print("✓ Done\n")
    
    # Evaluate all models
    print("\n" + "="*80)
    print("EVALUATING MODELS ON TEST SET")
    print("="*80 + "\n")
    
    results = []
    predictions = {}
    optimal_thresholds = {}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        
        # For neural network, use scaled data
        if name == 'Neural Network':
            metrics, y_pred_proba = evaluate_model(model, X_test_scaled, y_test, model_name=name)
        else:
            metrics, y_pred_proba = evaluate_model(model, X_test, y_test, model_name=name)
        
        predictions[name] = y_pred_proba
        optimal_thresholds[name] = metrics['optimal_f1_threshold']
        
        results.append({
            'model': name,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'optimal_threshold': metrics['optimal_f1_threshold'],
            'optimal_f1': metrics['optimal_f1_score']
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
    results_path = REPORTS_DIR / 'model_comparison.csv'
    results_df.to_csv(results_path, index=False)
    print(f"\n✓ Results saved to: {results_path}")
    
    # Detailed report for best model
    best_model_name = results_df.iloc[0]['model']
    best_threshold = optimal_thresholds[best_model_name]
    print_detailed_report(y_test, predictions[best_model_name], best_threshold, best_model_name)
    
    # Visualizations
    print("\n" + "="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80 + "\n")
    
    plot_roc_curves(predictions, y_test)
    plot_performance_comparison(results_df)
    plot_confusion_matrices(predictions, y_test, optimal_thresholds)
    
    # Feature importance for tree-based models
    if 'Random Forest' in models:
        feature_names = vectorizer.get_feature_names_out()
        importances = models['Random Forest'].feature_importances_
        plot_feature_importance(feature_names, importances, 'Random Forest', top_n=20)
    
    # Save best models
    print("\n" + "="*80)
    print("SAVING MODELS")
    print("="*80 + "\n")
    
    # Save top 3 models
    for idx, row in results_df.head(3).iterrows():
        model_name = row['model']
        model = models[model_name]
        
        model_path = MODELS_DIR / f"{model_name.lower().replace(' ', '_')}_model.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✓ Saved: {model_path}")
    
    # Save vectorizer and scaler
    with open(MODELS_DIR / 'vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(MODELS_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    print("\n✓ Vectorizer and scaler saved")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()