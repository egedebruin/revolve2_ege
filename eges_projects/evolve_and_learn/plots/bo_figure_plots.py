import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler

# Toy objective function (e.g., some performance measure over controller parameters)
def objective(x):
    return np.sin(3 * x) + x**2 - 0.7 * x

# Generate data for the true objective function
X = np.linspace(-2, 2, 400).reshape(-1, 1)
Y = objective(X)

# Sampled points (already evaluated)
X_samples = np.array([[-1.5], [-0.75] ,[1.5]])
Y_samples = objective(X_samples)

# Standardize input data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_samples_scaled = scaler.transform(X_samples)

# Define a kernel with Matern covariance function
kernel = Matern(length_scale=1.0, nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, n_restarts_optimizer=10, random_state=2)
gp.fit(X_samples_scaled, Y_samples)

# Predict mean and uncertainty (standard deviation)
Y_mean, Y_std = gp.predict(X_scaled, return_std=True)

# Plot the true objective function
plt.figure(figsize=(10, 6))

# Plot the GP mean and uncertainty
plt.plot(X, Y_mean, 'b-', label='GP mean', linewidth=5)
plt.fill_between(X.ravel(), Y_mean - 1.96 * Y_std, Y_mean + 1.96 * Y_std,
                 color='blue', alpha=0.2, label='Uncertainty (95% CI)')

# Plot sampled points
plt.plot(X_samples, Y_samples, 'ro', label='Evaluated samples', markersize=8)

# Labels and title
# plt.xlabel('Controller parameters', fontsize=22)
# plt.ylabel('Objective value', fontsize=22)
plt.title("Surrogate function", fontsize=36)
plt.xticks([])
plt.yticks([])
plt.show()

# Define the acquisition function (Upper Confidence Bound)
def acquisition_ucb(X, gp, kappa=2.0):
    mean, std = gp.predict(X, return_std=True)
    return mean + kappa * std

# Calculate UCB acquisition values
UCB = acquisition_ucb(X_scaled, gp)

# Plot the true objective function
plt.figure(figsize=(10, 6))

# Plot sampled points
plt.plot(X_samples, Y_samples, 'ro', label='Evaluated samples', markersize=8)

# Plot the UCB acquisition function
plt.plot(X, UCB, 'g-', label='UCB Acquisition', alpha=0.9, linewidth=5)

plt.axvline(x=-1.1, color='black', linestyle='--')

# Labels and title
# plt.xlabel('Controller parameters', fontsize=22)
# plt.ylabel('Objective value', fontsize=22)
plt.title("Acquisition function", fontsize=36)
plt.xticks([])
plt.yticks([])
plt.show()
