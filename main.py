from src.preprocessing import Preprocessing
from src.models import RFModel, XGBModel
from src.config import RF_PARAMS, XGB_PARAMS

if __name__ == "__main__":

    prep = (Preprocessing() # chaining: prima prende i dati, poi crea le feature, poi prepara i set di train/test con .get_data() e .get_full_data() per cross-validation
            .load_data("Dataset/raw/housing.csv")
            .feature_engineering()      
            .prepare_features())

    X_tr, y_tr, X_te, y_te = prep.get_data()
    X_full, y_full = prep.get_full_data()

    models = [
        RFModel("rf",  **RF_PARAMS),    
        XGBModel("xgb", **XGB_PARAMS),
    ]

    for m in models:
        m.cross_validate(X_full, y_full)  
        m.train(X_tr, y_tr).predict(X_te).evaluate(y_te)
        m.save()