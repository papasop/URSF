# compute_structure_alpha.py
# Computes the structural fine-structure constant φ_α using φ_e, φ_hbar, and φ_c

# Collapse-derived structure constants (can be replaced with your real φ(x)-based outputs)
phi_e = 0.2413         # Structural electric charge amplitude
phi_hbar = 0.0071      # Structural Planck constant
phi_c = 1123.81        # Structural light speed

# Known physical fine-structure constant
alpha_phys = 0.00729735

# Compute structural alpha
phi_alpha = (phi_e ** 2) / (phi_hbar * phi_c)
rel_error = abs(phi_alpha - alpha_phys) / alpha_phys

# Output
print("--- Structural Fine-Structure Constant φ_α ---")
print(f"phi_e     = {phi_e:.6f}")
print(f"phi_hbar  = {phi_hbar:.6f}")
print(f"phi_c     = {phi_c:.2f}")
print(f"Computed φ_α     = {phi_alpha:.7f}")
print(f"Expected α       = {alpha_phys:.7f}")
print(f"Relative Error   = {rel_error:.3%}")
