data {
    int<lower=0> N; // Number of observations
    int<lower=0> K; // Number of locations
    int<lower=0> M; // Number of months
    int<lower=0> cases[N]; // Disease cases
    int<lower=1,upper=K> location[N]; // Location for each observation
    int<lower=1,upper=M> month[N]; // Month for each observation
    real temperature[N]; // Temperature for each observation
}

parameters {
    real alpha; // Intercept
    real beta; // Slope for temperature
    real<lower=0> phi; // Overdispersion parameter for negative binomial
    vector[K] location_effect; // Random effect for location
    vector[M] month_effect; // Random effect for month
    real<lower=0> sigma_location; // Standard deviation for location effects
    real<lower=0> sigma_month; // Standard deviation for month effects
}

model {
    vector[N] lambda;

    // Priors
    alpha ~ normal(0, 10);
    beta ~ normal(0, 10);
    phi ~ normal(0, 10);
    location_effect ~ normal(0, sigma_location);
    month_effect ~ normal(0, sigma_month);
    sigma_location ~ normal(0, 10);
    sigma_month ~ normal(0, 10);

    // Linear predictor
    for (n in 1:N) {
        lambda[n] = exp(alpha + beta * temperature[n] + location_effect[location[n]] + month_effect[month[n]]);
    }

    // Negative binomial likelihood
    cases ~ neg_binomial_2(lambda, phi);
}

generated quantities {
    vector[N] log_lik;
    for (n in 1:N) {
        log_lik[n] = neg_binomial_2_lpmf(cases[n] | lambda[n], phi);
    }
}
