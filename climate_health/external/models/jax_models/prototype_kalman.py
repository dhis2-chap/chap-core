"""
z_forward[t] = Normal(mu_even[t], sigma_even[t])
y_hat[t] = H[t]@mu_prev[t]+D[t]u[t]+d[t]
S[t] = H[t]@sigma_prev[t]@H[t].T+ R[t]
K[t] = sigma_prev[t]@H[t].T@S[t]**-1
mu_even[t] = mu_prev[t]+K[t]@(y[t]-y_hat[t])
sigma[t] = sigma_prev[t]-K[t]@H[t]@sigma_prev[t]
sigma[t] = sigma_prev[t]-K[t]@S[t]@K[t].T
"""
