# Percorso dataset

"""
CONTENUTO DEL FILE:

contiene i parametri generali del progetto, come il percorso del dataset, 
le feature numeriche e categoriche, il target e i parametri per la divisione dei dati e la cross-validation.

le costanti in poche parole, per evitare di doverle riscrivere in più file e per avere un unico punto di riferimento


DATA_PATH = "Dataset/raw/housing.csv"
"""

# Features
NUM_FEATURES = [
    'longitude', 'latitude', 'housing_median_age',
    'total_rooms', 'total_bedrooms', 'population',
    'households', 'median_income'
]

# Parametri flessibili per tutti i modelli
RF_PARAMS = {
    "n_estimators": 200,
    "max_depth": 10,
    "random_state": 42
}

XGB_PARAMS = {
    "n_estimators": 300,
    "learning_rate": 0.1,
    "max_depth": 6,
    "random_state": 42
}

CAT_FEATURES = ['ocean_proximity']
TARGET = 'median_house_value'

# Parametri generali
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5