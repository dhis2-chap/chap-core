import mlflow

import mlflow

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from mlflow.models import validate_serving_input


class CustomModel(mlflow.pyfunc.PythonModel):
    def __init__(self, factor):
        self.factor = factor

    def load_context(self, context):
        # This method can be used to load model-specific artifacts if needed
        pass

    def predict(self, context, model_input):
        # This method defines the prediction logic
        print("Predicting, factor is ", self.factor)
        return model_input * self.factor



def store_model():
    model = CustomModel(2)
    data = load_diabetes()
    X = data.data
    y = data.target
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    signature = infer_signature(X_train, model.predict(None, X_train))
    mlflow.pyfunc.log_model("model", python_model=model, signature=signature, )


def load():
    logged_model = 'runs:/6917f831946b42b0bcfc4fd39f0329f4/model'
    data = load_diabetes()
    X = data.data
    y = data.target
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Predict on a Pandas DataFrame.
    import pandas as pd
    loaded_model.predict(X_train)


#train_and_log()
#run_stored_model()
#test()
load()
