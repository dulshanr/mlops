
import mlflow
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow import MlflowClient


mlflow.set_tracking_uri("http://127.0.0.1:5001")
client = MlflowClient()


model_uri = "models:/model_v2/1"
# mlflow.set_experiment("MLflow Quickstart")

print(mlflow.get_tracking_uri())

# for a in client.list_artifacts("6f4b10132063438ca016f4b412ff29d2"):
#     print(a.path)


X, y = datasets.load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

loaded_model = mlflow.pyfunc.load_model(model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])