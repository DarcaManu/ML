import pandas as pd
from src.config import DATA_PATH
from src.preprocessing import preparazione_dati
from src.models import REGRESSORS, PARAM_GRID_XGB
from src.evaluate import compare_models_split, compare_models_crossval, tune_xgboost

df = pd.read_csv(DATA_PATH)
X_train, X_test, y_train, y_test, preprocessor = preparazione_dati(df)
X_full = preprocessor.fit_transform(df.drop('median_house_value', axis=1))
y_full = df['median_house_value']

compare_models_split(REGRESSORS, X_train, X_test, y_train, y_test)
compare_models_crossval(REGRESSORS, X_full, y_full)
best_model = tune_xgboost(PARAM_GRID_XGB, X_full, y_full)