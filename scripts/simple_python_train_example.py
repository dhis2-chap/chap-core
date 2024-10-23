import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib


def train(csv_fn, model_fn):
    df = pd.read_csv(csv_fn)
    features = ['rainfall', 'mean_temperature']
    X = df[features]
    Y = df['disease_cases']
    Y = Y.fillna(0)  # set NaNs to zero (not a good solution, just for the example to work)
    model = LinearRegression()
    model.fit(X, Y)
    joblib.dump(model, model_fn)


train('example_data/v0/training_data.csv', 'model.pkl')


def predict(model_fn, historic_data_fn, future_climatedata_fn, predictions_fn):
    df = pd.read_csv(future_climatedata_fn)
    cols = ['rainfall', 'mean_temperature']
    X = df[cols]
    model = joblib.load(model_fn)

    predictions = model.predict(X)

    train_data = pd.read_csv(historic_data_fn)
    y_train = train_data['disease_cases']
    X_train = train_data[cols]

    # Estimate the residual variance from the training data
    residuals = y_train - model.predict(X_train)
    residual_variance = np.var(residuals)

    # Generate sampled predictions by adding Gaussian noise
    n_samples = 20  # Number of samples you want
    sampled_predictions = []

    for i in range(n_samples):
        noise = np.random.normal(0, np.sqrt(residual_variance), size=predictions.shape)

        # add the samples to the dataframe we write as output
        df[f'sample_{i}'] = predictions + noise

    df.to_csv(predictions_fn, index=False)


predict('model.pkl', 'example_data/v0/historic_data.csv', 'example_data/v0/future_data.csv', 'predictions.csv')
