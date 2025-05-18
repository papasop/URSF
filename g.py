import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# --- Collapse-path φ(x) structure ---
def phi_collapse(x, lambda0, theta0, delta_theta, A_base, C_norm, N=10, delta_lambda=1.0):
    return C_norm * sum(
        (A_base / (n + 1)**2) * np.cos((lambda0 + n * delta_lambda) * np.log(x) + theta0 + n * delta_theta)
        for n in range(N)
    )

# --- δ(x) residual function: difference from target constant ---
def delta_x(x, target_value, params):
    return phi_collapse(x, *params) - target_value

# --- H(t) entropy-like function: variance of modal projection over log(x) range ---
def spectral_entropy(x_vals, target_value, params):
    delta_vals = np.array([delta_x(x, target_value, params) for x in x_vals])
    return np.log(1 + delta_vals**2)  # Entropy proxy

# --- Constants and parameters for test ---
target_value = 1 / np.pi
x_vals = np.linspace(3.0, 7.0, 400)

# Use best-fit parameters from previous π⁻¹ collapse test
params_pi = [1880.0309330537061, 4936.4385981016785, 3651.2149096332355, 3003.872775539271, 783.6408475397579]

# Compute δ(x) and entropy H(x)
delta_vals = [delta_x(x, target_value, params_pi) for x in x_vals]
H_vals = spectral_entropy(x_vals, target_value, params_pi)

# Plot δ(x) and H(t) to evaluate x₀ = 5.0
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(x_vals, delta_vals, label='δ(x)')
plt.axvline(5.0, color='r', linestyle='--', label='x₀ = 5.0')
plt.title('Residual δ(x)')
plt.xlabel('x')
plt.ylabel('δ(x)')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(x_vals, H_vals, label='H(x) = log(1 + δ²)', color='orange')
plt.axvline(5.0, color='r', linestyle='--', label='x₀ = 5.0')
plt.title('Entropy Proxy H(x)')
plt.xlabel('x')
plt.ylabel('H(x)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
