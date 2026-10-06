from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'data.csv'
MODELS_DIR = BASE_DIR / 'results' / 'best_models'
PLOTS_DIR = BASE_DIR / 'results' / 'plots'
REPORTS_DIR = BASE_DIR / 'results' / 'reports'

# Create directories
for dir_path in [MODELS_DIR, PLOTS_DIR, REPORTS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.25

# Features
NUM_FEATURES = ['seniorcitizen', 'tenure', 'monthlycharges', 'totalcharges']
CATEGORICAL_FEATURES = [
    'gender', 'partner', 'dependents', 'phoneservice', 'multiplelines',
    'internetservice', 'onlinesecurity', 'onlinebackup', 'deviceprotection',
    'techsupport', 'streamingtv', 'streamingmovies', 'contract',
    'paperlessbilling', 'paymentmethod'
]

# Hyperparameter grids
HYPERPARAMETER_GRIDS = {
    'LogisticRegression': {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000],
        'class_weight': [None, 'balanced']
    },
    'DecisionTree': {
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 10, 20, 50, 100],
        'min_samples_leaf': [1, 2, 5, 10, 20],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    },
    'RandomForest': {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [10, 15, 20, 25, None],
        'min_samples_split': [2, 10, 50, 100],
        'min_samples_leaf': [1, 2, 5, 10],
        'max_features': ['sqrt', 'log2', None],
        'class_weight': [None, 'balanced']
    },
    'GradientBoosting': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 2, 5],
        'subsample': [0.7, 0.8, 0.9, 1.0]
    },
    'XGBoost': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9],
        'min_child_weight': [1, 3, 5, 7],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'gamma': [0, 0.1, 0.2, 0.3],
        'reg_alpha': [0, 0.1, 0.5, 1],
        'reg_lambda': [1, 1.5, 2]
    },
    'LightGBM': {
        'n_estimators': [50, 100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 5, 7, 9, -1],
        'num_leaves': [15, 31, 63, 127],
        'min_child_samples': [10, 20, 30, 50],
        'subsample': [0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
}

# Neural Network configuration
NN_CONFIG = {
    'hidden_layers': [128, 64, 32],
    'dropout_rates': [0.3, 0.3, 0.2],
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'patience': 15,
    'weight_decay': 1e-5
}

# Neural Network hyperparameter search
NN_HYPERPARAM_SEARCH = {
    'architectures': [
        {'layers': [64, 32], 'dropouts': [0.3, 0.2]},
        {'layers': [128, 64, 32], 'dropouts': [0.3, 0.3, 0.2]},
        {'layers': [256, 128, 64], 'dropouts': [0.4, 0.3, 0.2]},
        {'layers': [128, 64], 'dropouts': [0.3, 0.2]},
        {'layers': [256, 128, 64, 32], 'dropouts': [0.4, 0.3, 0.3, 0.2]}
    ],
    'learning_rates': [0.0001, 0.0005, 0.001, 0.005],
    'batch_sizes': [16, 32, 64],
    'weight_decays': [1e-6, 1e-5, 1e-4]
}

# Threshold optimization
THRESHOLD_RANGE = (0.1, 0.9)
THRESHOLD_STEPS = 81

# Cross-validation
CV_FOLDS = 5
CV_SCORING = 'roc_auc'
N_JOBS = -1

# Visualization
FIGURE_DPI = 300
PLOT_STYLE = 'whitegrid'