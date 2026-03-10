from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    news = request.form["news"]

    transformed = vectorizer.transform([news])

    prediction = model.predict(transformed)[0]

    prob = model.predict_proba(transformed)[0]

    confidence = round(np.max(prob)*100,2)

    result = "REAL" if prediction=="REAL" else "FAKE"

    return render_template(
        "result.html",
        prediction=result,
        confidence=confidence,
        news=news
    )

if __name__ == "__main__":
    app.run(debug=True)