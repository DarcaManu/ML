import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from src.config import CV_FOLDS, RANDOM_STATE

# CONTENUTO DEL FILE:
# contiene le funzioni per confrontare i modelli sia con la divisione classica in
# train/test che con la cross-validation, e una funzione per il tuning di XGBoost con GridSearchCV

def compare_models_split(regressors, X_train, X_test, y_train, y_test):
    print("SPLIT 80/20 - RMSE Test Set")
    results = {}
    for nome, config in regressors.items():
        model = config["model"](**config["params"])
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        results[nome] = rmse
        print(f"  {nome}: {rmse:.4f}")
    best = min(results, key=results.get)
    print(f"\nMigliore SPLIT: {best} (RMSE: {results[best]:.4f})")
    return results

def compare_models_crossval(regressors, X_full, y_full):
    print("CROSS-VALIDATION 5-fold - RMSE Medio")
    results = {}
    for nome, config in regressors.items():
        model = config["model"](**config["params"])
        scores = cross_val_score(
            model, X_full, y_full,
            cv=CV_FOLDS,
            scoring='neg_root_mean_squared_error'
        )
        rmse = -scores.mean()
        results[nome] = rmse
        print(f"  {nome}: {rmse:.4f}")
    best = min(results, key=results.get)
    print(f"\nMigliore CV: {best} (RMSE: {results[best]:.4f})")
    return results

def tune_xgboost(param_grid, X_full, y_full):
    print("Tuning XGBoost con GridSearchCV...")
    xgb = XGBRegressor(random_state=RANDOM_STATE)
    gs = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring='neg_root_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    gs.fit(X_full, y_full)
    print(f"\nBest params: {gs.best_params_}")
    print(f"Best RMSE:   {-gs.best_score_:.4f}")
    return gs.best_estimator_