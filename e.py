# collapse_phi_e_reconstruction_local.py
# Pure standalone script: collapse-path φ_e ≈ 0.2413 reconstruction at multiple x values
# No file dependencies, prints results directly

import numpy as np

# Parameters
N = 30
lambda_0 = 100.0
lambda_n = np.array([lambda_0 + i for i in range(N)])
x_values = [3.0, 4.0, 5.0, 6.0, 7.0]
phi_e_target = 0.2413
use_mixed_basis = True
align_phase = False

# Generate sin/cos basis at a given x
def generate_basis(x, lambda_n, align_phase=True, mixed_basis=True):
    if align_phase:
        theta_n = -lambda_n * np.log(x)
    else:
        theta_n = np.zeros_like(lambda_n)
    cos_terms = np.cos(lambda_n * np.log(x) + theta_n)
    if mixed_basis:
        sin_terms = np.sin(lambda_n * np.log(x) + theta_n)
        return np.concatenate([cos_terms, sin_terms])
    else:
        return cos_terms

# Solve amplitudes to match φ(x) ≈ φ_e_target
def fit_amplitudes(basis, target):
    norm_sq = np.sum(basis**2)
    if norm_sq == 0:
        return 0.0, 1.0
    A_opt = target * basis / norm_sq
    phi_e_out = np.sum(A_opt * basis)
    error = abs(phi_e_out - target)
    rel_error = error / target
    return phi_e_out, rel_error

# Main
print("--- Collapse-Path φ_e Reconstruction (Pure Local Version) ---")
for x in x_values:
    basis = generate_basis(x, lambda_n, align_phase=align_phase, mixed_basis=use_mixed_basis)
    phi_e_out, rel_error = fit_amplitudes(basis, phi_e_target)
    print(f"x = {x:.1f} → φ_e = {phi_e_out:.6f} | Relative Error = {rel_error:.3%}")
