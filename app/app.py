from flask import Flask, request, jsonify
import pickle
import os

app = Flask(__name__)

# Load model from local file (inside Docker image)
MODEL_PATH = "model/model.pkl"

model = None

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    raise Exception("Model file not found at model/model.pkl")

@app.route("/", methods=["GET"])
def health():
    return jsonify({
        "status": "running",
        "model_loaded": model is not None
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    features = [
        data["sepal length (cm)"],
        data["sepal width (cm)"],
        data["petal length (cm)"],
        data["petal width (cm)"]
    ]

    prediction = model.predict([features])[0]

    return jsonify({
        "prediction": int(prediction)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080)
