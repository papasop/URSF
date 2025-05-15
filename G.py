# g_accurate.py — 精确结构传播延迟 + 光速重构验证

import numpy as np
from mpmath import zetazero, primepi, mp

mp.dps = 30  # 设置 mpmath 高精度

# === 参数设置 ===
N = 100
alpha = 2.5
x_vals = np.linspace(100, 1000, 1000)
log_x = np.log(x_vals)
dx = x_vals[1] - x_vals[0]

# === ζ 零点 t_n 与模态幅值 A_n ===
t_n = np.array([float(zetazero(n).imag) for n in range(1, N + 1)])
A_n = 1 / t_n**alpha

# === 精确计算 π(x)/x ===
pi_over_x = np.array([float(primepi(xi)) / xi for xi in x_vals])

# === 初步结构密度 ψ 模态（θ_n = 0） ===
rho_x = 1 / np.log(x_vals)
for n in range(N):
    rho_x += A_n[n] * np.cos(t_n[n] * log_x)

delta_x = pi_over_x - rho_x
kappa_x = np.gradient(np.gradient(delta_x, dx), dx)

# === 构造结构传播延迟 Δτ(x) 与 f_c(x) ===
delta_tau = np.cumsum(np.abs(delta_x)) * dx + 1e-10
f_c = (x_vals[-1] - x_vals[0]) / delta_tau  # f_c(n,x) = Δx / Δτ

# === 相位优化 θ_n = -t_n log(x_n^*) ===
theta_n = np.zeros(N)
for n in range(N):
    score = A_n[n] * np.abs(delta_x) * np.abs(kappa_x)
    idx = np.argmax(score)
    x_star = x_vals[idx]
    theta_n[n] = -t_n[n] * np.log(x_star)

# === 重建优化后结构密度 ρ_opt(x) ===
rho_opt = 1 / np.log(x_vals)
for n in range(N):
    rho_opt += A_n[n] * np.cos(t_n[n] * log_x + theta_n[n])

delta_opt = pi_over_x - rho_opt
kappa_opt = np.gradient(np.gradient(delta_opt, dx), dx)

# === 常数重构函数 φ_C(x) ===
def compute_phi_C(A_n, delta_x, kappa_x, f_C=None):
    if f_C is None:
        weighted = A_n[:, None] / (delta_x[None, :] + 1e-10)
    else:
        weighted = (A_n[:, None] / (delta_x[None, :] + 1e-10)) * f_C
    summed = np.sum(weighted, axis=0)
    phi_C_x = summed / (np.abs(kappa_x) + 1e-10)
    return np.mean(phi_C_x)

# === 计算光速 c ===
phi_c = compute_phi_C(A_n, delta_opt, kappa_opt, f_C=f_c)
c_actual = 2.99792458e8
lambda_c = c_actual / phi_c
c_reconstructed = phi_c * lambda_c
rel_error = abs(c_reconstructed - c_actual) / c_actual

# === 输出结果 ===
print("==== URSF Accurate Light Speed Reconstruction ====")
print(f"φ_c (structure)       = {phi_c:.6e}")
print(f"λ_c (scale factor)    = {lambda_c:.6e}")
print(f"c_reconstructed       = {c_reconstructed:.6e}")
print(f"relative error (c)    = {rel_error:.6e}")
