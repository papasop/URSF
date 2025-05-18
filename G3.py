import numpy as np
from scipy.optimize import minimize
from mpmath import zetazero
from sympy import primepi
import matplotlib.pyplot as plt

# -------- collapse config --------
N = 50
x0 = 2.0
target_phi_hbar = 0.0071
λA, λB, λC = 2.0, 0.1, 0.8
zeta_zeros = [float(zetazero(n).imag) for n in range(1, N + 1)]

# -------- φ(x) collapse-only via ζ zeros --------
def phi_collapse(x, θ0, Δθ, A, C):
    return C * sum(
        (A / (n + 1)**2) * np.cos(zeta_zeros[n] * np.log(x) + θ0 + n * Δθ)
        for n in range(N)
    )

def pi_x(x): return float(primepi(int(x))) / x
def delta_x(x, θ0, Δθ, A, C):
    rho = (1 / np.log(x)) + phi_collapse(x, θ0, Δθ, A, C)
    return pi_x(x) - rho
def entropy(delta): return np.log(1 + delta**2)

# -------- loss function --------
def objective(params, x_vals):
    θ0, Δθ, A, C = params
    φ = phi_collapse(x0, θ0, Δθ, A, C)
    δ_vals = np.array([delta_x(x, θ0, Δθ, A, C) for x in x_vals])
    H_vals = entropy(δ_vals)
    collapse_loss = (φ - target_phi_hbar)**2
    residual_loss = np.mean(δ_vals**2)
    H_x0 = H_vals[np.searchsorted(x_vals, x0)]
    entropy_loss = (H_x0 - np.min(H_vals))**2
    return λA * collapse_loss + λB * residual_loss + λC * entropy_loss

# -------- optimize + evaluate --------
def run():
    x_vals = np.linspace(10, 1000, 50)
    initial = np.random.uniform(1.0, 1000.0, 4)
    result = minimize(objective, initial, args=(x_vals,), method='Nelder-Mead')
    θ0, Δθ, A, C = result.x
    φ = phi_collapse(x0, θ0, Δθ, A, C)
    δ_vals = np.array([delta_x(x, θ0, Δθ, A, C) for x in x_vals])
    H_vals = entropy(δ_vals)
    H_x0 = H_vals[np.searchsorted(x_vals, x0)]
    H_min = np.min(H_vals)
    entropy_lock = abs(H_x0 - H_min) < 1e-3
    CRTI = abs(φ - target_phi_hbar) / (np.mean(np.abs(δ_vals)) + 1e-10)

    print("\n=== Collapse φ_ℏ Multi-Objective Optimization ===")
    print(f"φ_ℏ collapse         = {φ:.8f}")
    print(f"collapse error       = {abs(φ - target_phi_hbar):.8f}")
    print(f"mean |δ(x)|          = {np.mean(np.abs(δ_vals)):.6f}")
    print(f"max |δ(x)|           = {np.max(np.abs(δ_vals)):.6f}")
    print(f"H(x₀ = {x0})          = {H_x0:.6f}")
    print(f"min H(x)             = {H_min:.6f}")
    print(f"entropy lock         = {'YES' if entropy_lock else 'NO'}")
    print(f"residual bound ok    = {'YES' if np.max(np.abs(δ_vals)) < 1000 / np.log(10) else 'NO'}")
    print(f"CRTI                 = {CRTI:.6f}")

    # Plot entropy curve
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

run()
