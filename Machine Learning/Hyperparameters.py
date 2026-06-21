import pandas as pd
import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
import warnings

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

def load_and_prepare_data():
    df = pd.read_csv('/.csv')
    if 'Tt' in df.columns:
        df = df.drop(columns=['Tt'])
        
    features = [col for col in df.columns if col not in ['label', 'trajectory_label']]
    X = df[features]
    y = df['label']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=True)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    return X_train_scaled, y_train

X_train, y_train = load_and_prepare_data()

cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
print(f"Data loaded successfully! Training feature shape: {X_train.shape}, 10-Fold CV configured.")


def objective_rf(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 1e-2, log=True),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42,
        'n_jobs': 1
    }
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1_macro').mean()
    return score

def objective_dt(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 10, 100),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-5, 1e-2, log=True),
        'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
        'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
        'random_state': 42
    }
    model = DecisionTreeClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1_macro').mean()
    return score

def objective_knn(trial):
    params = {
        'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
        'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
        'algorithm': trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree']),
        'leaf_size': trial.suggest_int('leaf_size', 10, 100),
        'p': trial.suggest_int('p', 1, 4),
        'n_jobs': 1
    }
    model = KNeighborsClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1_macro').mean()
    return score

print("Lightweight models (RF, DT, KNN) objective functions defined.")


N_TRIALS_FAST = 15 

fast_tasks = {
    "Decision Tree (DT)": objective_dt,
    "Random Forest (RF)": objective_rf,
    "K-Nearest Neighbors (KNN)": objective_knn
}

print("="*60)
print(" 🚀 Starting Lightweight Models Hyperparameter Tuning ")
print("="*60)

for model_name, objective_func in fast_tasks.items():
    print(f"\nTuning hyperparameters for {model_name} ({N_TRIALS_FAST} trials)...")
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_func, n_trials=N_TRIALS_FAST, n_jobs=-1) 
    
    best_params = study.best_params
    print(f"🎯 {model_name} Best 10-Fold Macro F1: {study.best_value:.4f}")
    print("-" * 40)
    
    prefix = model_name.split(" ")[-1].replace("(", "").replace(")", "")
    for key, value in best_params.items():
        var_name = f"{prefix}_{key.upper()}"
        if isinstance(value, str):
            print(f"{var_name} = '{value}'")
        else:
            print(f"{var_name} = {value}")
    print("-" * 40)


def objective_mlp(trial):
    layers = trial.suggest_categorical('hidden_layer_sizes', [(64,32), (64, 16)])
    params = {
        'hidden_layer_sizes': layers,
        'activation': trial.suggest_categorical('activation', ['relu', 'tanh', 'logistic']),
        'solver': trial.suggest_categorical('solver', ['adam', 'sgd']),
        'alpha': trial.suggest_float('alpha', 1e-4, 1e-3, log=True),
        'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-3, 1e-2, log=True),
        'max_iter': trial.suggest_int('max_iter', 110, 150),
        'early_stopping': True,
        'random_state': 42
    }
    model = MLPClassifier(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1_macro').mean()
    return score

def objective_svm(trial):
    params = {
        'C': trial.suggest_float('C', 100.0, 1000.0, log=True),
        'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
        'gamma': trial.suggest_categorical('gamma', 0.1, 1.0, log=True),
        'random_state': 42
    }
    model = SVC(**params)
    score = cross_val_score(model, X_train, y_train, cv=cv_strategy, scoring='f1_macro').mean()
    return score


N_TRIALS_SLOW = 10 

slow_tasks = {
    "Multi-Layer Perceptron (MLP)": objective_mlp,
    "Support Vector Machine (SVM)": objective_svm
}

print("="*60)
print(" ⏳ Starting Computationally Intensive Models Hyperparameter Tuning ")
print("="*60)

for model_name, objective_func in slow_tasks.items():
    print(f"\nTuning hyperparameters for {model_name} ({N_TRIALS_SLOW} trials)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective_func, n_trials=N_TRIALS_SLOW, n_jobs=2) 
    
    print(f"🎯 {model_name} Best 10-Fold Macro F1: {study.best_value:.4f}")
    prefix = model_name.split(" ")[-1].replace("(", "").replace(")", "")
    for key, value in study.best_params.items():
        var_name = f"{prefix}_{key.upper()}"
        print(f"{var_name} = '{value}'" if isinstance(value, str) else f"{var_name} = {value}")