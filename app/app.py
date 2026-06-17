from flask import Flask, request, jsonify
import mlflow.pyfunc
import pandas as pd

app = Flask(__name__)

MODEL_URI = "mlruns/0/models/m-1a7b1e21508b4aa791e6aa1de6ba80f0/artifacts"

model = mlflow.pyfunc.load_model(MODEL_URI)

@app.route("/")
def home():
    return jsonify({
        "status": "running",
        "model_loaded": True
    })

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    features = pd.DataFrame([data])

    prediction = model.predict(features)

    return jsonify({
        "prediction": int(prediction[0])
    })

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=8080
    )
