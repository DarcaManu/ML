# Model Selection Project

Progetto di regressione sul dataset California Housing.
Confronto tra modelli, cross-validation e tuning con GridSearchCV.

---

## Struttura del progetto, Apri dal file per vedere meglio

model_selection/
├── main.py
├── README.md
├── requirements.txt
├── Dataset/
│   └── raw/
│       └── housing.csv
├── models/
├── notebooks/
│   └── Model_selection.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── preprocessing.py
    ├── models.py
    └── evaluate.py

---

## src/config.py

Variabili globali del progetto. Modifica qui i parametri senza toccare il resto del codice.

| Variabile      | Tipo  | Descrizione                                      |
|----------------|-------|--------------------------------------------------|
| DATA_PATH      | str   | Percorso relativo al dataset CSV                 |
| NUM_FEATURES   | list  | Lista delle feature numeriche                    |
| CAT_FEATURES   | list  | Lista delle feature categoriche                  |
| TARGET         | str   | Nome della colonna target                        |
| RANDOM_STATE   | int   | Seed per riproducibilita'                        |
| TEST_SIZE      | float | Percentuale dati di test (default 0.2)           |
| CV_FOLDS       | int   | Numero di fold per cross-validation (default 5)  |

---

## src/preprocessing.py

Preparazione e trasformazione dei dati grezzi.

### preparazione_dati(df)

Esegue il preprocessing completo del DataFrame e restituisce i dati pronti per il training.

Parametri:
- df — DataFrame pandas con i dati grezzi

Restituisce:
- X_train     — dati di training
- X_test      — dati di test
- y_train     — target di training
- y_test      — target di test
- preprocessor — oggetto ColumnTransformer gia' fittato

Operazioni interne:
- Crea feature engineered: rooms_per_household, bedrooms_per_room, population_per_household
- Imputa i valori NaN con la mediana sulle feature numeriche
- Scala le feature numeriche con StandardScaler
- Applica OneHotEncoding sulla feature categorica ocean_proximity
- Splitta i dati in train/test secondo TEST_SIZE e RANDOM_STATE

Uso:
    from src.preprocessing import preparazione_dati
    X_train, X_test, y_train, y_test, preprocessor = preparazione_dati(df)

---

## src/models.py

Definizione dei modelli e dei parametri per la GridSearch.

### REGRESSORS

Dizionario con tutti i modelli da confrontare.

| Chiave            | Modello                | Parametri principali              |
|-------------------|------------------------|-----------------------------------|
| LinearRegression  | LinearRegression       | nessuno                           |
| DecisionTree      | DecisionTreeRegressor  | max_depth=5                       |
| RandomForest      | RandomForestRegressor  | n_estimators=100                  |
| XGBoost           | XGBRegressor           | n_estimators=100, learning_rate=0.1 |

Uso:
    from src.models import REGRESSORS

    for nome, config in REGRESSORS.items():
        model = config["model"](**config["params"])

Per aggiungere un modello:
    "Ridge": {
        "model": Ridge,
        "params": {"alpha": 1.0}
    }

---

### PARAM_GRID_XGB

Griglia di parametri usata da tune_xgboost() per la GridSearchCV su XGBoost.

| Parametro        | Valori            |
|------------------|-------------------|
| n_estimators     | [200, 300, 400]   |
| learning_rate    | [0.08, 0.1, 0.12] |
| max_depth        | [5, 6, 7]         |
| subsample        | [0.85, 0.9]       |
| colsample_bytree | [0.85, 0.9]       |
| reg_alpha        | [0, 0.1]          |
| reg_lambda       | [1, 1.5]          |

---

## src/evaluate.py

Funzioni per valutare e confrontare i modelli.

### compare_models_split(regressors, X_train, X_test, y_train, y_test)

Confronta tutti i modelli del dizionario usando lo split 80/20.
Stampa l'RMSE di ciascuno e il migliore.

Parametri:
- regressors            — dizionario modelli (usa REGRESSORS da models.py)
- X_train, X_test       — dati di training e test
- y_train, y_test       — target di training e test

Restituisce:
- results — dizionario {nome_modello: rmse}

Uso:
    from src.evaluate import compare_models_split
    results = compare_models_split(REGRESSORS, X_train, X_test, y_train, y_test)

---

### compare_models_crossval(regressors, X_full, y_full)

Confronta tutti i modelli usando cross-validation a 5 fold.
Piu' affidabile dello split singolo.

Parametri:
- regressors — dizionario modelli (usa REGRESSORS da models.py)
- X_full     — dati completi preprocessati
- y_full     — target completo

Restituisce:
- results — dizionario {nome_modello: rmse_medio}

Uso:
    from src.evaluate import compare_models_crossval
    results = compare_models_crossval(REGRESSORS, X_full, y_full)

---

### tune_xgboost(param_grid, X_full, y_full)

Esegue GridSearchCV su XGBoost per trovare la combinazione ottimale di iperparametri.

Parametri:
- param_grid — dizionario parametri (usa PARAM_GRID_XGB da models.py)
- X_full     — dati completi preprocessati
- y_full     — target completo

Restituisce:
- best_estimator_ — modello XGBoost gia' fittato con i parametri migliori

Uso:
    from src.evaluate import tune_xgboost
    best_model = tune_xgboost(PARAM_GRID_XGB, X_full, y_full)

---

## Esecuzione

Da terminale:
    python3 main.py

Da notebook:
    jupyter notebook
    Apri notebooks/Model_selection.ipynb ed esegui le celle in ordine.

---

## Dipendenze

manualmente:
    pip install requirements.txt
