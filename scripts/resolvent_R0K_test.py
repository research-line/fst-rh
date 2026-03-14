#!/usr/bin/env python3
"""
resolvent_R0K_test.py
=====================
Tests Part B of the M1'' strategy:
Compute ||R_0 K|| for the sin and cos blocks to verify that
diagonal resolvent is a good approximation.

R_0 = diag(1/gap_j) (diagonal resolvent)
K = off-diagonal part of QW block
||R_0 K|| = operator norm of R_0 @ K

If ||R_0 K|| < 1, Neumann series converges and diagonal PT2 ≈ true PT2.
"""
import numpy as np
from scipy.linalg import eigh, norm
from sympy import primerange
import sys

LOG4PI_GAMMA_F = 3.2720532309274587


def build_basis_grid(N, t_grid, L, basis="cos"):
    phi = np.zeros((N, len(t_grid)))
    if basis == "cos":
        phi[0, :] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, :] = np.cos(n * np.pi * t_grid / L) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, :] = np.sin((n + 1) * np.pi * t_grid / L) / np.sqrt(L)
    return phi


def build_shifted_basis(N, t_grid, L, shift, basis="cos"):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    if basis == "cos":
        phi[0, mask] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, mask] = np.cos(n * np.pi * ts[mask] / L) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, mask] = np.sin((n + 1) * np.pi * ts[mask] / L) / np.sqrt(L)
    return phi


def build_QW(lam, N, primes, basis="cos", n_quad=2000, n_int=1000):
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = build_basis_grid(N, t_grid, L, basis)
    W = LOG4PI_GAMMA_F * np.eye(N)
    s_max = min(2 * L, 12.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        Kval = np.exp(s / 2) / (2.0 * np.sinh(s))
        if Kval < 1e-15:
            continue
        pp = build_shifted_basis(N, t_grid, L, s, basis)
        pm = build_shifted_basis(N, t_grid, L, -s, basis)
        Sp = (phi @ pp.T) * dt
        Sm = (phi @ pm.T) * dt
        W += Kval * (Sp + Sm - 2.0 * np.exp(-s / 2) * np.eye(N)) * ds
    for p in primes:
        logp = np.log(p)
        for m in range(1, 20):
            coeff = logp * p ** (-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            pp = build_shifted_basis(N, t_grid, L, shift, basis)
            pm = build_shifted_basis(N, t_grid, L, -shift, basis)
            Sp = (phi @ pp.T) * dt
            Sm = (phi @ pm.T) * dt
            W += coeff * (Sp + Sm)
    return W


if __name__ == "__main__":
    print("=" * 80)
    print("||R_0 K|| TEST: Diagonal Resolvent Approximation Quality")
    print("=" * 80)
    sys.stdout.flush()

    lambdas = [100, 200, 500, 1000, 2000, 3000]
    N_sin = 25
    N_cos = 10

    print(f"\n{'lam':>6s} | {'||R0K||_sin':>12s} | {'||R0K||_cos':>12s} | {'PT2_err_sin':>12s} | {'PT2_err_cos':>12s} | {'gap_ratio':>10s}")
    print("-" * 80)

    for lam in lambdas:
        L = np.log(lam)
        primes = [int(p) for p in primerange(2, lam + 1)]

        # Build full matrices
        Wc = build_QW(lam, N_cos, primes, "cos")
        Ws = build_QW(lam, N_sin, primes, "sin")

        # === SIN SECTOR ===
        # Leading mode: n=0, gaps relative to W_00
        W00_sin = Ws[0, 0]
        # Diagonal gaps
        diag_gaps_sin = np.array([Ws[j, j] - W00_sin for j in range(1, N_sin)])
        # Off-diagonal: coupling of mode 0 to others
        # Full off-diagonal matrix K (zero diagonal)
        K_sin = Ws.copy()
        np.fill_diagonal(K_sin, 0)

        # R_0 for sin: diagonal with 1/gap_j for j>=1, and 0 for j=0
        R0_sin = np.zeros_like(Ws)
        for j in range(1, N_sin):
            if abs(diag_gaps_sin[j-1]) > 1e-15:
                R0_sin[j, j] = 1.0 / diag_gaps_sin[j-1]

        R0K_sin = R0_sin @ K_sin
        r0k_norm_sin = norm(R0K_sin, 2)  # operator norm

        # True eigenvalue vs PT2 for sin
        evals_sin = np.sort(eigh(Ws, eigvals_only=True))
        l1_sin = evals_sin[0]
        sigma_sin = l1_sin - W00_sin  # true coupling shift
        # PT2 estimate
        B_sin = np.array([Ws[0, j] for j in range(1, N_sin)])
        E_sin = np.sum(B_sin**2 / diag_gaps_sin)
        pt2_err_sin = abs(abs(sigma_sin) - E_sin) / abs(sigma_sin) if abs(sigma_sin) > 1e-10 else 0

        # === COS SECTOR ===
        W11_cos = Wc[1, 1]
        # Diagonal gaps: j != 1
        cos_indices = [j for j in range(N_cos) if j != 1]
        diag_gaps_cos = np.array([Wc[j, j] - W11_cos for j in cos_indices])

        K_cos = Wc.copy()
        np.fill_diagonal(K_cos, 0)

        R0_cos = np.zeros_like(Wc)
        for idx, j in enumerate(cos_indices):
            if abs(diag_gaps_cos[idx]) > 1e-15:
                R0_cos[j, j] = 1.0 / diag_gaps_cos[idx]

        R0K_cos = R0_cos @ K_cos
        r0k_norm_cos = norm(R0K_cos, 2)

        evals_cos = np.sort(eigh(Wc, eigvals_only=True))
        l1_cos = evals_cos[0]
        sigma_cos = l1_cos - W11_cos
        B_cos = np.array([Wc[1, j] for j in cos_indices])
        E_cos = np.sum(B_cos**2 / diag_gaps_cos)
        pt2_err_cos = abs(abs(sigma_cos) - E_cos) / abs(sigma_cos) if abs(sigma_cos) > 1e-10 else 0

        # Gap ratio: min(diag_gap) / max(off_diag) for sin
        gap_ratio = np.min(np.abs(diag_gaps_sin)) / np.max(np.abs(B_sin)) if np.max(np.abs(B_sin)) > 0 else 0

        print(f"{lam:6d} | {r0k_norm_sin:12.6f} | {r0k_norm_cos:12.6f} | {pt2_err_sin:12.4f} | {pt2_err_cos:12.4f} | {gap_ratio:10.4f}")
        sys.stdout.flush()

    print("\n" + "=" * 80)
    print("INTERPRETATION:")
    print("  ||R_0 K|| < 1  =>  Neumann series converges")
    print("  ||R_0 K|| << 1 =>  Diagonal PT2 ≈ true PT2")
    print("  PT2 error should decrease with lambda")
    print("=" * 80)
