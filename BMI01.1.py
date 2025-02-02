# Importare le librerie necessarie
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Impostiamo un seme per la riproducibilità
np.random.seed(42)

# Numero di campioni nel dataset
num_samples = 500

# Generazione dati casuali
peso = np.random.uniform(50, 120, num_samples)  # Peso tra 50 e 120 kg
altezza = np.random.uniform(1.50, 2.00, num_samples)  # Altezza tra 1.50 e 2.00 m
eta = np.random.randint(18, 65, num_samples)  # Età tra 18 e 65 anni
sesso = np.random.choice([0, 1], num_samples)  # 0 = Femmina, 1 = Maschio

# Calcolo del BMI
bmi = peso / (altezza ** 2)

# Creazione DataFrame
df = pd.DataFrame({
    'Peso (kg)': peso,
    'Altezza (m)': altezza,
    'Età': eta,
    'Sesso': sesso,
    'BMI': bmi
})

# Stampiamo le prime 5 righe per controllo
print("Prime 5 righe del dataset:")
print(df.head())

# Salviamo il dataset in un file CSV
df.to_csv("dataset_bmi.csv", index=False)
print("\n✅ Dataset salvato correttamente come 'dataset_bmi.csv'!\n")

# Carichiamo il dataset
df = pd.read_csv("dataset_bmi.csv")

# Controlliamo le informazioni
print("\n🔍 Informazioni sul dataset:")
print(df.info())

# Controlliamo se ci sono valori mancanti
print("\n🔍 Valori mancanti nel dataset:")
print(df.isnull().sum())

# Separare le variabili indipendenti (X) e la variabile target (y)
X = df[['Peso (kg)', 'Altezza (m)', 'Età', 'Sesso']]
y = df['BMI']

# Dividere il dataset in training set (80%) e test set (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Controlliamo la dimensione dei dati
print("\n📊 Dimensioni del dataset dopo la suddivisione:")
print(f"Training Set: {X_train.shape}, {y_train.shape}")
print(f"Test Set: {X_test.shape}, {y_test.shape}")

# Creare e addestrare il modello di Machine Learning
model = LinearRegression()
model.fit(X_train, y_train)

print("\n✅ Modello addestrato con successo!")

# Fare previsioni
y_pred = model.predict(X_test)

# Stampiamo le prime 5 previsioni e i valori reali
print("\n🔍 Confronto tra previsioni e valori reali:")
print(f"Previsioni: {y_pred[:5]}")
print(f"Valori reali: {y_test[:5].values}")

# Calcoliamo le metriche di errore
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

print("\n📊 Valutazione del modello:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

# Salviamo il modello per utilizzarlo in Flask
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("\n✅ Modello salvato correttamente in 'model.pkl'!")
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd

# Carichiamo il dataset
df = pd.read_csv("dataset_bmi.csv")

# Definiamo le variabili indipendenti (X) e la variabile target (y)
X = df[['Peso (kg)', 'Altezza (m)', 'Età', 'Sesso']]
y = df['BMI']

# Creiamo e addestriamo il modello
model = LinearRegression()
model.fit(X, y)

# Salviamo il modello in un file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Modello salvato correttamente in 'model.pkl'!")
import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd

# Carichiamo il dataset
df = pd.read_csv("dataset_bmi.csv")

# Definiamo le variabili indipendenti (X) e la variabile target (y)
X = df[['Peso (kg)', 'Altezza (m)', 'Età', 'Sesso']]
y = df['BMI']

# Creiamo e addestriamo il modello
model = LinearRegression()
model.fit(X, y)

# Salviamo il modello in un file
with open("bmi_app/model.pkl", "wb") as file:
    pickle.dump(model, file)

print("✅ Modello salvato correttamente in 'bmi_app/model.pkl'!")



