from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

from src.config import NUM_FEATURES, CAT_FEATURES, TARGET, TEST_SIZE, RANDOM_STATE #prendiamo dagli altri file i parametri che ci servono

import pandas as pd

""""
CONTENUTO DEL FILE:
ColumnTransformer + il feature engineering
"""


def preparazione_dati(df):
    df['rooms_per_household']       = df['total_rooms']    / df['households']
    df['bedrooms_per_room']         = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household']  = df['population']     / df['households']

    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), NUM_FEATURES),
        ('cat', Pipeline([
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ]), CAT_FEATURES)
    ])

    X_processed = preprocessor.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test, preprocessor