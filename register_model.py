import pickle
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier()

model.fit(X, y)

with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path="model"
    )

print("Model registered to MLflow")
