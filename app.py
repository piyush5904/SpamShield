from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained SpamShield model
model = joblib.load("spamshield_model.pkl")


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    confidence = None

    if request.method == "POST":

        message = request.form["message"]

        prediction = model.predict([message])[0]

        probabilities = model.predict_proba([message])[0]

        # Get probability of the predicted class
        confidence = probabilities[list(model.classes_).index(prediction)]

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence
    )


if __name__ == "__main__":
    app.run(debug=True)