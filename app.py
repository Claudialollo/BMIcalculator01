from flask import Flask, render_template, request
import pickle
import numpy as np

# Creiamo l'app Flask
app = Flask(__name__, template_folder="templates")

# Carichiamo il modello salvato
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Funzione per convertire peso e altezza
def convert_units(peso, unita_peso, altezza, unita_altezza):
    if unita_peso == "lbs":
        peso = peso * 0.453592  # Converti lbs in kg
    if unita_altezza == "cm":
        altezza = altezza / 100  # Converti cm in metri
    return peso, altezza


# Pagina principale con il form HTML
@app.route("/", methods=["GET", "POST"])
def index():
    bmi_pred = None
    if request.method == "POST":
        try:
            # Prendiamo i dati dal form
            peso = float(request.form["peso"])
            altezza = float(request.form["altezza"])
            eta = int(request.form["eta"])
            sesso = int(request.form["sesso"])

            # Creiamo un array con i dati dell'utente
            dati_utente = np.array([[peso, altezza, eta, sesso]])

            # Facciamo la previsione
            bmi_pred = model.predict(dati_utente)[0]
        except Exception as e: 
            bmi_pred = None 

    return render_template("index.html", bmi=bmi_pred)

# Esegui l'app Flask


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))  # Usa la porta di Render, altrimenti 5000
    app.run(host="0.0.0.0", port=port)

