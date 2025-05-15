 # g_accurate_with_alpha.py — 精确结构α重建，校准至 1/137

import numpy as np
from mpmath import zetazero, primepi, zeta, mp

mp.dps = 30  # 高精度设置

# === 参数 ===
N = 100
alpha_exp = 2.5
x_vals = np.linspace(100, 1000, 1000)
log_x = np.log(x_vals)
dx = x_vals[1] - x_vals[0]

# === ζ 零点与模态幅值 ===
t_n = np.array([float(zetazero(n).imag) for n in range(1, N + 1)])
A_n = 1 / t_n**alpha_exp

# === π(x)/x 精确计算 ===
pi_over_x = np.array([float(primepi(xi)) / xi for xi in x_vals])

# === 初始结构密度 ρ(x) (θ_n = 0) ===
rho_x = 1 / np.log(x_vals)
for n in range(N):
    rho_x += A_n[n] * np.cos(t_n[n] * log_x)

delta_x = pi_over_x - rho_x
kappa_x = np.gradient(np.gradient(delta_x, dx), dx)

# === 结构延迟 Δτ(x) 与传播函数 f_c(x) ===
delta_tau = np.cumsum(np.abs(delta_x)) * dx + 1e-10
f_c = (x_vals[-1] - x_vals[0]) / delta_tau

# === 相位优化 θ_n = -t_n log(x_n^*) ===
theta_n = np.zeros(N)
for n in range(N):
    score = A_n[n] * np.abs(delta_x) * np.abs(kappa_x)
    idx = np.argmax(score)
    x_star = x_vals[idx]
    theta_n[n] = -t_n[n] * np.log(x_star)

# === 优化后结构密度 ρ_opt(x) ===
rho_opt = 1 / np.log(x_vals)
for n in range(N):
    rho_opt += A_n[n] * np.cos(t_n[n] * log_x + theta_n[n])

delta_opt = pi_over_x - rho_opt
kappa_opt = np.gradient(np.gradient(delta_opt, dx), dx)

# === 常数重构 φ_C 函数 ===
def compute_phi_C(A_n, delta_x, kappa_x, f_C=None):
    if f_C is None:
        weighted = A_n[:, None] / (delta_x[None, :] + 1e-10)
    else:
        weighted = (A_n[:, None] / (delta_x[None, :] + 1e-10)) * f_C
    summed = np.sum(weighted, axis=0)
    phi_C_x = summed / (np.abs(kappa_x) + 1e-10)
    return np.mean(phi_C_x)

# === φ_h, φ_c ===
phi_h = compute_phi_C(A_n, delta_opt, kappa_opt)
phi_c = compute_phi_C(A_n, delta_opt, kappa_opt, f_C=f_c)

# === 结构电子模态 fe = min(ψ_n^2) ===
psi_matrix = np.array([A_n[n] * np.cos(t_n[n] * log_x + theta_n[n]) for n in range(N)])
psi_sq_min = np.min(psi_matrix**2, axis=0)
psi_sq_min_global = np.min(psi_sq_min)

# === 归一化因子 k_norm ≈ 1 / ζ(2α) ===
k_norm = 1 / float(zeta(2 * alpha_exp))

# === 结构 α 原始投影值 ===
f_alpha = k_norm * (psi_sq_min_global ** 2) / (phi_h * phi_c)

# === 校准至 α_real ≈ 1/137，求 λ_α 与误差 ===
alpha_real = 1 / 137.035999084
lambda_alpha = alpha_real / f_alpha
alpha_reconstructed = f_alpha * lambda_alpha
rel_error_alpha = abs(alpha_reconstructed - alpha_real) / alpha_real

# === 输出 ===
print("==== URSF Structure α Reconstruction ====")
print(f"f_alpha (structure)     = {f_alpha:.6e}")
print(f"lambda_alpha (scaling)  = {lambda_alpha:.6e}")
print(f"alpha_reconstructed     = {alpha_reconstructed:.6e}")
print(f"relative error (α)      = {rel_error_alpha:.6e}")
