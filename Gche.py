# g_ursf_struct_alpha.py — 本地结构常数与 α 重构验证（无校准版本）

import numpy as np
from mpmath import zetazero, primepi, zeta, mp

mp.dps = 30  # 高精度

# === 参数设置 ===
N = 100
alpha_exp = 2.5
x_vals = np.linspace(100, 1000, 1000)
log_x = np.log(x_vals)
dx = x_vals[1] - x_vals[0]

# === ζ 零点 & 模态幅值 ===
t_n = np.array([float(zetazero(n).imag) for n in range(1, N + 1)])
A_n = 1 / t_n**alpha_exp

# === 精确 π(x)/x ===
pi_over_x = np.array([float(primepi(xi)) / xi for xi in x_vals])
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
    x_star = x_vals[np.argmax(score)]
    theta_n[n] = -t_n[n] * np.log(x_star)

# === 优化结构密度 ρ_opt(x) ===
rho_opt = 1 / np.log(x_vals)
for n in range(N):
    rho_opt += A_n[n] * np.cos(t_n[n] * log_x + theta_n[n])
delta_opt = pi_over_x - rho_opt
kappa_opt = np.gradient(np.gradient(delta_opt, dx), dx)

# === 通用 φ_C 计算函数 ===
def compute_phi_C(A_n, delta_x, kappa_x, f_C=None):
    if f_C is None:
        weighted = A_n[:, None] / (delta_x[None, :] + 1e-10)
    else:
        weighted = (A_n[:, None] / (delta_x[None, :] + 1e-10)) * f_C
    phi_C_x = np.sum(weighted, axis=0) / (np.abs(kappa_x) + 1e-10)
    return np.mean(phi_C_x)

# === 结构常数计算 ===
phi_h = compute_phi_C(A_n, delta_opt, kappa_opt)
phi_c = compute_phi_C(A_n, delta_opt, kappa_opt, f_C=f_c)
phi_G = compute_phi_C(A_n, delta_opt, kappa_opt)

# === ψ 模态最小激活：平均模态极小值方式 ===
psi_matrix = np.array([A_n[n] * np.cos(t_n[n] * log_x + theta_n[n]) for n in range(N)])
psi_sq_min_avg = np.mean(np.min(psi_matrix**2, axis=1))

# === k_norm ≈ 1/ζ(2α)
k_norm = 1 / float(zeta(2 * alpha_exp))
f_alpha = k_norm * (psi_sq_min_avg ** 2) / (phi_h * phi_c)

# === Γ_activation 定义（结构压缩因子）
Gamma_activation = 1.0 / np.min(psi_sq_min_avg * np.abs(delta_x) * np.abs(delta_tau))
lambda_alpha = Gamma_activation / (phi_h * phi_c)
alpha_reconstructed = f_alpha * lambda_alpha

# === 电子激活能（φ_e）在中区间
mid_mask = (x_vals >= 300) & (x_vals <= 700)
phi_e = np.min(psi_matrix[:, mid_mask]**2)

# === 输出结构打印结果 ===
print("==== URSF Structural Constants and α (Uncalibrated) ====")
print(f"φ_G  = {phi_G:.6e}")
print(f"φ_h  = {phi_h:.6e}")
print(f"φ_c  = {phi_c:.6e}")
print(f"φ_e  = {phi_e:.6e}")
print("\n--- Structure α Path ---")
print(f"f_alpha (structural)    = {f_alpha:.6e}")
print(f"Γ_activation (compression) = {Gamma_activation:.6e}")
print(f"λ_alpha (structural)    = {lambda_alpha:.6e}")
print(f"α_reconstructed         = {alpha_reconstructed:.6e}")
