# Model Selection - California Housing

Progetto di machine learning sul dataset California Housing.
L'obiettivo è confrontare più modelli di regressione, valutarli con cross-validation e fare tuning su XGBoost.

---

## Com'è organizzato il progetto

```
ML/
├── main.py
├── README.md
├── requirements.txt
├── Dataset/
│   └── raw/
│       └── housing.csv
├── models/          <- i modelli addestrati vengono salvati qui come .pkl
├── notebooks/
│   └── Model_selection.ipynb
└── src/
    ├── __init__.py
    ├── config.py
    ├── preprocessing.py
    ├── models.py
    └── compareModels.py
```

---

## Cosa fa ogni file

**`src/config.py`**
Contiene tutti i parametri globali (path del dataset, parametri dei modelli, numero di fold, ecc.).
Ho messo tutto qui così se voglio cambiare qualcosa non devo toccare il resto del codice.

**`src/preprocessing.py`**
Si occupa di caricare i dati, fare feature engineering e preparare i set di train/test.
Ho creato alcune feature derivate come `rooms_per_hh`, `bedrooms_per_room` e `pop_per_hh` perché le variabili originali da sole non catturavano bene la struttura dei dati.
Ho aggiunto anche un clustering geografico con KMeans su latitudine, longitudine e reddito — l'idea è che la zona geografica influisce molto sul prezzo e il modello da solo fatica a coglierlo.
La classe è strutturata con method chaining quindi in `main.py` si può scrivere tutto in una riga.

**`src/models.py`**
Contiene una classe astratta `ModelBase` con i metodi comuni (train, predict, evaluate, cross_validate, save, load) e due classi figlie: `RFModel` per RandomForest e `XGBModel` per XGBoost.
Ho usato l'ereditarietà perché i due modelli hanno la stessa interfaccia e così evito di ripetere codice.

**`src/compareModels.py`**
Funzioni per confrontare più modelli sia con split classico 80/20 che con cross-validation a 5 fold.
C'è anche `TuningXGBoost` che usa GridSearchCV e `BayesianOptimizationXGBoost` che usa scikit-optimize — la bayesiana è più efficiente perché non prova tutte le combinazioni ma guida la ricerca in base ai risultati precedenti.

**`main.py`**
Esegue tutto il flusso: preprocessing → cross-validation → train → evaluate → salvataggio modello.

---

## Come eseguirlo

Installa le dipendenze:
```bash
pip install -r requirements.txt
```

Avvia da terminale:
```bash
python3 main.py
```

Oppure apri il notebook:
```bash
jupyter notebook
# poi apri notebooks/Model_selection.ipynb
```

---

## Scelte che ho fatto e perché

- **Ho rimosso i prezzi >= 500.000$** dal dataset perché sono un valore artificiale di cap, non case reali
- **Ho usato `fit_transform` solo sul train e `transform` sul test** per evitare data leakage (se standardizzo anche con i dati di test, il modello "vede" informazioni che non dovrebbe avere)
- **Cross-validation invece del singolo split** perché con un solo split il risultato dipende da come casca la divisione, con 5 fold è più stabile
- **XGBoost come modello principale per il tuning** perché nei confronti iniziali era quello con RMSE più basso
