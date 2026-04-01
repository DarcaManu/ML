from abc import ABC, abstractmethod  # ABC: classe astratta, abstractmethod: forza override nelle figlie
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib   # Salva/carica modelli binari (.pkl)
import os

class ModelBase(ABC):
    """Classe astratta: definisce interfaccia comune a tutti i modelli"""
    def __init__(self, model_name, **kwargs):  # ← Aggiungi **kwargs
        self.name = model_name
        self.params = kwargs  # ← Aggiungi questa riga
        self.model = None
        self.y_pred_train = self.y_pred_test = None
        self.metrics = {}

    @abstractmethod
    def _build_model(self):
        """Ogni classe figlia DEVE implementare questo metodo"""
        pass

    def train(self, X_train, y_train):
        """Crea + addestra modello (polimorfico: funziona per RF e XGB)"""
        self.model = self._build_model()    # Chiama implementazione specifica
        self.model.fit(X_train, y_train)    # API sklearn universale
        self.y_pred_train = self.model.predict(X_train)
        print(f"{self.name} addestrato su {X_train.shape[0]} campioni")
        return self  # Permette chaining: .train().predict().evaluate()

    def predict(self, X_test):
        """Predice su test set"""
        self.y_pred_test = self.model.predict(X_test)
        return self

    def evaluate(self, y_test):
        """Calcola RMSE e R² e li salva in self.metrics"""
        rmse = mean_squared_error(y_test, self.y_pred_test) ** 0.5  # radice manuale
        r2 = r2_score(y_test, self.y_pred_test)
        self.metrics = {'rmse': rmse, 'r2': r2}
        print(f"{self.name}: RMSE={rmse:.1f}, R²={r2:.3f}")
        return self

    def save(self, path_prefix="models/"):
        """Salva modello addestrato su disco come file .pkl"""
        os.makedirs(path_prefix, exist_ok=True)  # Crea cartella models/ se non esiste
        joblib.dump(self.model, f"{path_prefix}{self.name}.pkl")
        print(f"Modello salvato: {self.name}.pkl")

# Classi concrete: ereditano tutto da ModelBase, implementano solo _build_model
    def _build_model(self):
        return RandomForestRegressor(**self.params, random_state=42)  # ← self.params contiene i parametri specifici per RF (es. n_estimators, max_depth)



class XGBModel(ModelBase):
    def _build_model(self):
        return xgb.XGBRegressor(**self.params, random_state=42)  # Gradient boosting