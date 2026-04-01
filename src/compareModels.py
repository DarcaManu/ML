import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBRegressor
from src.config import CV_FOLDS, RANDOM_STATE

# CONTENUTO DEL FILE:
# contiene le funzioni per confrontare i modelli sia con la divisione classica in
# train/test che con la cross-validation, e una funzione per il tuning di XGBoost con GridSearchCV

def comparatoreModelSplit(regressors, X_train, X_test, y_train, y_test):

    print("SPLIT 80/20 - RMSE Test Set")
    results = {}

    for nome, config in regressors.items():# config è un dizionario con chiavi "model" e "params"

        #questa linea di codice crea un'istanza del modello specificato in config["model"] (es. RandomForestRegressor) mettendo i parametri specificati nella sezione del dizionario "params"
        model = config["model"](**config["params"])
        model.fit(X_train, y_train)#addestra il modello

        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))#mostra dopo il valore dell RMSE
        results[nome] = rmse
        print(f"  {nome}: {rmse:.4f}")# stampa nome modello e RMSE con 4 decimali
    
    best = min(results, key=results.get)# trova il nome del modello con il RMSE più basso (migliore) e lo stampa
    print(f"\nMigliore SPLIT: {best} (RMSE: {results[best]:.4f})")

    return results

def comparatoreModelCrossval(regressors, X_full, y_full):
    print("CROSS-VALIDATION 5-fold - RMSE Medio")
    results = {}

    for nome, config in regressors.items():
        model = config["model"](**config["params"])

        scores = cross_val_score(
            model, X_full, y_full,
            cv=CV_FOLDS,# numero di fold per la cross-validation, definito in config.py
            scoring='neg_root_mean_squared_error'#RMSE negativo perchè cross_val_score meglio così (massimizza invece di minimizzare)
        )

        rmse = -scores.mean()# calcola RMSE medio (negativo da cross_val_score) e lo salva nei risultati
        results[nome] = rmse
        print(f"  {nome}: {rmse:.4f}")

    best = min(results, key=results.get)
    print(f"\nMigliore CV: {best} (RMSE: {results[best]:.4f})")
    
    return results

def TuningXGBoost(param_grid, X_full, y_full):

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

def BaesyanOptimizationXGBoost(X_full, y_full):
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer

    print("Tuning XGBoost con ottimizzazione bayesiana...")
    xgb = XGBRegressor(random_state=RANDOM_STATE)

    searchSpaces = {
    'n_estimators':      Integer(200, 1000),#variamo il numero di alberi per vedere se migliora la performance
    'learning_rate':     Real(0.01, 0.2, prior='log-uniform'),#variamo il learning rate per vedere se migliora la performance
    'max_depth':         Integer(3, 9),
    'subsample':         Real(0.5, 1.0),
    'colsample_bytree':  Real(0.3, 1.0),
    'min_child_weight':  Integer(1, 10),
    'reg_alpha':         Real(1e-5, 10.0, prior='log-uniform'),
    'reg_lambda':        Real(1e-5, 10.0, prior='log-uniform'),
    }


    opt = BayesSearchCV(#opt è il nostro ottimizzatore bayesiano che cercherà i migliori iperparametri 
    xgb,
    search_spaces=searchSpaces,
    n_iter=40,          # 40 combinazioni smart 
    cv=5,
    scoring='neg_root_mean_squared_error',
    random_state=42,
    n_jobs=-1,
    verbose=1
    )

    opt.fit(X_full, y_full)

    print(f"\nBest params: {opt.best_params_}")
    print(f"Best RMSE:   {-opt.best_score_:.4f}")

    return opt.best_params_ 