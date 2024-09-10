import pandas as pd
import stan as pystan

# Load the data
training_data = pd.read_csv("training_data.csv")

# Prepare data for Stan
stan_data = {
    'N': len(training_data),
    'K': training_data['location'].nunique(),
    'M': training_data['month'].nunique(),
    'cases': training_data['cases'].values,
    'location': training_data['location'].values,
    'month': training_data['month'].values,
    'temperature': training_data['temperature'].values
}

# Compile the Stan model
model_code = open("model.stan", "r").read()
sm = pystan.model.build(program_code=model_code, data=stan_data)

# Fit the model
fit = sm.sampling(data=stan_data, iter=1000, chains=4)

# Save the fit object
import pickle
with open("disease_model_fit.pkl", "wb") as f:
    pickle.dump(fit, f)
