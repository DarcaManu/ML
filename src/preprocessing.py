import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class Preprocessing:
    def __init__(self, target_col='median_house_value', test_size=0.2, random_state=42):
        # target_col: colonna da predire | test_size: 20% dati per test | random_state: seme fisso
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()  # Normalizzatore riutilizzabile
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.df = None

    def load_data(self, filepath):
        """Legge CSV e applica pulizia base"""
        self.df = pd.read_csv(filepath)

        # Rimuovi outliers: prezzi >= 500000 sono anomali (valore cap del dataset)
        self.df = self.df[self.df['median_house_value'] < 500000].reset_index(drop=True)
        # reset_index: dopo drop righe, rigenera indice 0,1,2... senza buchi

        # Rimuovi colonna categorica: ocean_proximity non numerica, scaler la rifiuta
        self.df = self.df.drop('ocean_proximity', axis=1)

        # Sostituisce NaN con mediana di ogni colonna (robusto agli outliers)
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)

        print(f"Dataset caricato: {self.df.shape[0]} righe, {self.df.shape[1]} colonne")
        return self  # Permette chaining: .load_data().prepare_features()

    def prepare_features(self):
        """Split train/test + scaling"""
        if self.df is None:
            raise ValueError("Chiama prima load_data!")

        # Separa features (X) dal target (y)
        X = self.df.drop(self.target_col, axis=1)  # axis=1 = colonne
        y = self.df[self.target_col]

        # Split 80% train / 20% test con seme fisso per riproducibilità
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        # fit_transform su train: calcola media/std e trasforma
        # transform su test: usa STESSA media/std del train (no data leakage!)
        self.X_train_proc = self.scaler.fit_transform(self.X_train)
        self.X_test_proc = self.scaler.transform(self.X_test)

        print(f"Features pronte: train={self.X_train_proc.shape}, test={self.X_test_proc.shape}")
        return self

    def get_data(self):
        """Getter: ritorna tuple pronta per il training"""
        # .values converte Pandas Series → NumPy array (richiesto da sklearn)
        return self.X_train_proc, self.y_train.values, self.X_test_proc, self.y_test.values
    
    # Aggiungi questo metodo alla classe Preprocessing
    def get_full_data(self):
        """Restituisce dati completi preprocessati per tuning/CV"""
        return self.X_full, self.y_full