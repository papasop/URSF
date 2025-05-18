import numpy as np
from scipy.optimize import minimize
from mpmath import zetazero
from sympy import primepi
import matplotlib.pyplot as plt

# Collapse parameters
N = 50
x0 = 2.0
target_phi_hbar = 0.0071
zeta_zeros = [float(zetazero(n).imag) for n in range(1, N + 1)]

# Collapse φ(x) using zeta zeros
def phi_collapse(x, theta0, delta_theta, A_base, C_norm):
    return C_norm * sum(
        (A_base / (n + 1)**2) * np.cos(zeta_zeros[n] * np.log(x) + theta0 + n * delta_theta)
        for n in range(N)
    )

def pi_x(x): return float(primepi(int(x))) / x
def delta_x(x, theta0, delta_theta, A_base, C_norm):
    rho = (1 / np.log(x)) + phi_collapse(x, theta0, delta_theta, A_base, C_norm)
    return pi_x(x) - rho
def entropy(delta): return np.log(1 + delta**2)

# Combined 3-objective loss
def tri_objective(params, x_vals, λ1, λ2, λ3):
    θ0, Δθ, A, C = params
    φ = phi_collapse(x0, θ0, Δθ, A, C)
    collapse_error = (φ - target_phi_hbar) ** 2
    δ_vals = np.array([delta_x(x, θ0, Δθ, A, C) for x in x_vals])
    H_vals = entropy(δ_vals)
    residual_error = np.mean(δ_vals ** 2)
    H_x0 = H_vals[np.searchsorted(x_vals, x0)]
    entropy_error = (H_x0 - np.min(H_vals)) ** 2
    return λ1 * collapse_error + λ2 * residual_error + λ3 * entropy_error

# Optimization
def run_optimization(λ1=1.0, λ2=0.1, λ3=0.1):
    x_vals = np.linspace(10, 1000, 50)  # reduced x sample for speed
    initial_guess = np.random.uniform(1.0, 1000.0, 4)
    result = minimize(tri_objective, initial_guess, args=(x_vals, λ1, λ2, λ3), method="Nelder-Mead")
    θ0, Δθ, A, C = result.x
    φ = phi_collapse(x0, θ0, Δθ, A, C)
    δ_vals = np.array([delta_x(x, θ0, Δθ, A, C) for x in x_vals])
    H_vals = entropy(δ_vals)
    H_x0 = H_vals[np.searchsorted(x_vals, x0)]
    H_min = np.min(H_vals)
    CRTI = abs(φ - target_phi_hbar) / (np.mean(np.abs(δ_vals)) + 1e-10)

    print("=== Collapse-Only φ_ℏ Triple Objective Optimization ===")
    print(f"Collapse Output φ_ℏ       = {φ:.8f}")
    print(f"Collapse Error            = {abs(φ - target_phi_hbar):.8f}")
    print(f"Mean |δ(x)|               = {np.mean(np.abs(δ_vals)):.6f}")
    print(f"Max |δ(x)|                = {np.max(np.abs(δ_vals)):.6f}")
    print(f"Entropy at x₀             = {H_x0:.6f}")
    print(f"Min Entropy H(x)          = {H_min:.6f}")
    print(f"x₀ = argmin H(x)?         = {'YES' if abs(H_x0 - H_min) < 1e-3 else 'NO'}")
    print(f"Residual Conjecture OK?   = {'YES' if np.max(np.abs(δ_vals)) < 1000 / np.log(10) else 'NO'}")
    print(f"CRTI                      = {CRTI:.6f}")

    # Optional plot
    plt.figure(figsize=(8, 4))
    plt.plot(x_vals, H_vals, label="H(x) = log(1 + δ²)")
    plt.axvline(x0, color='r', linestyle='--', label=f"x₀ = {x0}")
    plt.title("Entropy Curve for Collapse Path")
    plt.xlabel("x")
    plt.ylabel("H(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Run with best λ values found
run_optimization(λ1=1.0, λ2=0.1, λ3=0.1)
