import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans


class Preprocessing:
    def __init__(self, target_col='median_house_value', test_size=0.2,
                 random_state=42, n_clusters=5):
        # target_col: colonna da predire | test_size: 20% dati per test | random_state: seme fisso
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()  # Normalizzatore riutilizzabile
        self.imputer = SimpleImputer(strategy='median')
        self.kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=random_state)
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.df = None

    def load_data(self, filepath):
        """Legge CSV e applica pulizia base"""
        self.df = pd.read_csv(filepath)

        # Rimuovi outliers: prezzi >= 500000 sono anomali (valore cap del dataset)
        self.df = self.df[self.df['median_house_value'] < 500000].reset_index(drop=True)
        # reset_index: dopo drop righe, rigenera indice 0,1,2... senza buchi

        print(f"Dataset caricato: {self.df.shape[0]} righe, {self.df.shape[1]} colonne")
        return self  # Permette chaining: .load_data().feature_engineering().prepare_features()

    def feature_engineering(self):
        """Crea feature derivate e clustering geografico — logica portata dal notebook"""
        if self.df is None:
            raise ValueError("Chiama prima load_data!")

        df = self.df

        #il median income non è lineare sul prezzo, se una famiglia guadagna 10k in più non è detto che possa
        # permettersi una casa che costa 10k
        df['income_squared'] = df['median_income'] ** 2

        # interazione posizione + reddito (Lezione 5: Interaction Features)
        df['income_x_lat'] = df['median_income'] * df['latitude']
        df['income_x_long'] = df['median_income'] * df['longitude']

        #ora invece faremo delle feature delle variabili più deboli per migliorare la performance
        df['rooms_per_hh'] = df['total_rooms'] / df['households']  # numero di stanze per famiglia
        df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']  # percentuale di stanze da letto rispetto al totale delle stanze
        df['pop_per_hh'] = df['population'] / df['households']  # numero di persone per famiglia
        df['income_per_room'] = df['median_income'] / df['total_rooms']  # costo per stanza, più è alto più è probabile che la casa sia costosa

        # Clustering geografico: cattura pattern non lineari tra zona e reddito
        cols_for_clustering = ['longitude', 'latitude', 'median_income']
        X_km = StandardScaler().fit_transform(df[cols_for_clustering])
        df['cluster'] = self.kmeans.fit_predict(X_km)  # applico il clustering ai dati standardizzati e creo una nuova colonna 'cluster' con i risultati del clustering

        # praticamente possiamo separarli molto facilmente con un one hot encoding
        df = pd.get_dummies(df, columns=['ocean_proximity'], drop_first=False)

        self.df = df
        print(f"Feature engineering completato: {self.df.shape[1]} colonne")
        return self  # Permette chaining

    def prepare_features(self):
        """Split train/test + imputation + scaling"""
        if self.df is None:
            raise ValueError("Chiama prima load_data!")

        # Separa features (X) dal target (y)
        X = self.df.drop(self.target_col, axis=1)  # axis=1 = colonne
        y = self.df[self.target_col]

        # Split 80% train / 20% test con seme fisso per riproducibilità
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state)

        # imputa NaN (total_bedrooms ha 207 valori mancanti, 1% del dataset)
        X_tr = self.imputer.fit_transform(self.X_train)
        X_te = self.imputer.transform(self.X_test)

        # fit_transform su train: calcola media/std e trasforma
        # transform su test: usa STESSA media/std del train (no data leakage!)
        self.X_train_proc = self.scaler.fit_transform(X_tr)
        self.X_test_proc = self.scaler.transform(X_te)

        print(f"Features pronte: train={self.X_train_proc.shape}, test={self.X_test_proc.shape}")
        return self

    def get_data(self):
        """Getter: ritorna tuple pronta per il training"""
        # .values converte Pandas Series → NumPy array (richiesto da sklearn)
        return self.X_train_proc, self.y_train.values, self.X_test_proc, self.y_test.values

    def get_full_data(self):
        """Restituisce dati completi preprocessati per tuning/CV"""
        X_full = np.vstack([self.X_train_proc, self.X_test_proc])
        y_full = np.concatenate([self.y_train.values, self.y_test.values])
        return X_full, y_full