from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
import numpy as np
from config import HYPERPARAMETER_GRIDS, CV_FOLDS, CV_SCORING, N_JOBS, RANDOM_STATE

def train_all_traditional_models(X_train, y_train, verbose=True):
    """Train all traditional ML models with default parameters."""
    
    models = {}
    
    if verbose:
        print("\n" + "="*80)
        print("TRAINING TRADITIONAL MODELS")
        print("="*80 + "\n")
    
    # Logistic Regression
    if verbose:
        print("Training Logistic Regression...")
    lr_model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train, y_train)
    models['Logistic Regression'] = lr_model
    if verbose:
        print("✓ Done\n")
    
    # Decision Tree
    if verbose:
        print("Training Decision Tree...")
    dt_model = DecisionTreeClassifier(max_depth=10, min_samples_split=50, random_state=RANDOM_STATE)
    dt_model.fit(X_train, y_train)
    models['Decision Tree'] = dt_model
    if verbose:
        print("✓ Done\n")
    
    # Random Forest
    if verbose:
        print("Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_split=50, 
                                      random_state=RANDOM_STATE, n_jobs=N_JOBS)
    rf_model.fit(X_train, y_train)
    models['Random Forest'] = rf_model
    if verbose:
        print("✓ Done\n")
    
    # Gradient Boosting
    if verbose:
        print("Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                         random_state=RANDOM_STATE)
    gb_model.fit(X_train, y_train)
    models['Gradient Boosting'] = gb_model
    if verbose:
        print("✓ Done\n")
    
    # XGBoost
    if verbose:
        print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                  random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=N_JOBS)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    if verbose:
        print("✓ Done\n")
    
    # LightGBM
    if verbose:
        print("Training LightGBM...")
    lgb_model = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, 
                                   random_state=RANDOM_STATE, verbose=-1, n_jobs=N_JOBS)
    lgb_model.fit(X_train, y_train)
    models['LightGBM'] = lgb_model
    if verbose:
        print("✓ Done\n")
    
    return models

def optimize_traditional_model(model_name, X_train, y_train, n_iter=20, verbose=True):
    """Optimize a traditional model using GridSearchCV with RandomizedSearchCV fallback."""
    
    if verbose:
        print("\n" + "="*80)
        print(f"OPTIMIZING {model_name.upper()}")
        print("="*80 + "\n")
    
    # Get model and parameter grid
    if model_name == 'LogisticRegression':
        base_model = LogisticRegression(random_state=RANDOM_STATE)
    elif model_name == 'DecisionTree':
        base_model = DecisionTreeClassifier(random_state=RANDOM_STATE)
    elif model_name == 'RandomForest':
        base_model = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=N_JOBS)
    elif model_name == 'GradientBoosting':
        base_model = GradientBoostingClassifier(random_state=RANDOM_STATE)
    elif model_name == 'XGBoost':
        base_model = xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=N_JOBS)
    elif model_name == 'LightGBM':
        base_model = lgb.LGBMClassifier(random_state=RANDOM_STATE, verbose=-1, n_jobs=N_JOBS)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    param_grid = HYPERPARAMETER_GRIDS.get(model_name, {})
    
    if not param_grid:
        if verbose:
            print(f"No hyperparameter grid defined for {model_name}")
        return base_model.fit(X_train, y_train), None
    
    # Calculate total combinations
    from itertools import product
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    
    if verbose:
        print(f"Total possible combinations: {total_combinations}")
    
    # Use RandomizedSearchCV if too many combinations
    if total_combinations > 100:
        from sklearn.model_selection import RandomizedSearchCV
        if verbose:
            print(f"Using RandomizedSearchCV with {n_iter} iterations\n")
        
        search = RandomizedSearchCV(
            base_model,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring=CV_SCORING,
            n_jobs=N_JOBS,
            random_state=RANDOM_STATE,
            verbose=1 if verbose else 0
        )
    else:
        if verbose:
            print(f"Using GridSearchCV with {total_combinations} combinations\n")
        
        search = GridSearchCV(
            base_model,
            param_grid=param_grid,
            cv=StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE),
            scoring=CV_SCORING,
            n_jobs=N_JOBS,
            verbose=1 if verbose else 0
        )
    
    # Fit
    search.fit(X_train, y_train)
    
    if verbose:
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        print(f"Best Score: {search.best_score_:.4f}")
        print(f"Best Parameters:")
        for param, value in search.best_params_.items():
            print(f"  {param}: {value}")
        print("="*80 + "\n")
    
    return search.best_estimator_, search.cv_results_