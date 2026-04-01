from src.preprocessing import Preprocessing
from src.models_oop import RFModel, XGBModel
from src.config import RF_PARAMS
from src.evaluate import BaesyanOptimizationXGBoost  # Funzione bayesiana

if __name__ == "__main__":

    # STEP 1: Preprocessing
    prep = (Preprocessing()
            .load_data('Dataset/raw/housing.csv')
            .prepare_features())
    X_tr, y_tr, X_te, y_te = prep.get_data()
    
    # Dati completi per tuning
    X_full, y_full = prep.get_full_data()

    # STEP 2: Tuning Bayesiano XGBoost
    print("\nTuning Bayesiano XGBoost...")
    best_xgb_params = BaesyanOptimizationXGBoost(X_full, y_full)

    # STEP 3: Modelli con parametri ottimali
    models = [
        RFModel('rf', **RF_PARAMS),
        XGBModel('xgb', **best_xgb_params)  # Parametri trovati automaticamente!
    ]
    results = {}

    # STEP 4: Training + valutazione
    for model in models:
        model.train(X_tr, y_tr)\
             .predict(X_te)\
             .evaluate(y_te)
        results[model.name] = model.metrics
        model.save()

    # STEP 5: Confronto
    print("\nCONFRONTO:")
    for name, mets in results.items():
        print(f"{name}: R²={mets['r2']:.3f}")

    best = max(results.items(), key=lambda x: x[1]['r2'])
    print(f"\nMIGLIORE: {best[0]} (R²={best[1]['r2']:.3f})")