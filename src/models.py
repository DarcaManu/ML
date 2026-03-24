from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from src.config import RANDOM_STATE

# CONTENUTO DEL FILE:
# definisce i modelli da utilizzare e i parametri per la grid search di XGBoost, 
# in modo da avere un unico punto di riferimento per i modelli e i loro parametri

REGRESSORS = {
    "LinearRegression": {
        "model": LinearRegression,
        "params": {}
    },
    "DecisionTree": {
        "model": DecisionTreeRegressor,
        "params": {"max_depth": 5, "random_state": RANDOM_STATE}
    },
    "RandomForest": {
        "model": RandomForestRegressor,
        "params": {"n_estimators": 100, "random_state": RANDOM_STATE}
    },
    "XGBoost": {
        "model": XGBRegressor,
        "params": {"n_estimators": 100, "learning_rate": 0.1, "random_state": RANDOM_STATE}
    }
}

PARAM_GRID_XGB = {
    'n_estimators':   [200, 300, 400],
    'learning_rate':  [0.08, 0.1, 0.12],
    'max_depth':      [5, 6, 7],
    'subsample':      [0.85, 0.9],
    'colsample_bytree': [0.85, 0.9],
    'reg_alpha':      [0, 0.1],
    'reg_lambda':     [1, 1.5]
}