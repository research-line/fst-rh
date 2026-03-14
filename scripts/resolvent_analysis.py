#!/usr/bin/env python3
"""
resolvent_analysis.py
=====================
Detailed resolvent-damped energy analysis for M1' characterization.

Computes:
1. E_sin, E_cos, D, and their ratios for a dense lambda grid
2. Common-mode cancellation: inner product of b_sin, b_cos vectors
3. Gap comparison: gap_j^sin vs gap_j^cos for each j
4. Scaling fits for (E_sin - E_cos)/D

Answers Copilot's question: gaps are diagonal differences, not eigenvalue gaps.
"""
import numpy as np
from scipy.linalg import eigh, norm
from sympy import primerange
import sys
import json

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
        K = np.exp(s / 2) / (2.0 * np.sinh(s))
        if K < 1e-15:
            continue
        pp = build_shifted_basis(N, t_grid, L, s, basis)
        pm = build_shifted_basis(N, t_grid, L, -s, basis)
        Sp = (phi @ pp.T) * dt
        Sm = (phi @ pm.T) * dt
        W += K * (Sp + Sm - 2.0 * np.exp(-s / 2) * np.eye(N)) * ds
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
    print("RESOLVENT-DAMPED ENERGY ANALYSIS FOR M1'")
    print("=" * 80)
    sys.stdout.flush()

    # Dense lambda grid
    lambdas = [50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
    N_cos = 10   # Even block
    N_sin = 25   # Odd block

    results = []

    for lam in lambdas:
        L = np.log(lam)
        primes = [int(p) for p in primerange(2, lam + 1)]

        Wc = build_QW(lam, N_cos, primes, "cos")
        Ws = build_QW(lam, N_sin, primes, "sin")

        # Leading diagonals
        W11_cos = Wc[1, 1]   # leading even mode
        W00_sin = Ws[0, 0]   # leading odd mode
        D = abs(W11_cos - W00_sin)  # Shift Parity diagonal advantage

        # Off-diagonal coupling vectors
        # Even: B_cos[j] = Wc[1, j] for j != 1
        # Odd:  B_sin[j] = Ws[0, j] for j >= 1
        B_cos = np.array([Wc[1, j] for j in range(N_cos) if j != 1])
        B_sin = np.array([Ws[0, j] for j in range(1, N_sin)])

        # Diagonal gaps (conservative: diagonal differences, not eigenvalue gaps)
        gap_cos = np.array([Wc[j, j] - W11_cos for j in range(N_cos) if j != 1])
        gap_sin = np.array([Ws[j, j] - W00_sin for j in range(1, N_sin)])

        # Resolvent-damped energies
        E_cos = np.sum(B_cos**2 / gap_cos)
        E_sin = np.sum(B_sin**2 / gap_sin)

        # True eigenvalues for comparison
        evals_cos = np.sort(eigh(Wc, eigvals_only=True))
        evals_sin = np.sort(eigh(Ws, eigvals_only=True))
        l1_cos = evals_cos[0]
        l1_sin = evals_sin[0]

        # True coupling differentials
        sigma_cos = l1_cos - W11_cos
        sigma_sin = l1_sin - W00_sin

        # PT2 accuracy
        pt2_accuracy = abs(E_sin - E_cos) / abs(sigma_sin - sigma_cos) * 100 if abs(sigma_sin - sigma_cos) > 1e-10 else 0

        # Certified gap
        cert_gap = l1_cos - l1_sin

        # Common-mode cancellation analysis
        # Normalize: b_sin = B_sin/sqrt(gap_sin), b_cos = B_cos/sqrt(gap_cos)
        # Need to match dimensions: take first min(len(B_cos), len(B_sin)) entries
        n_common = min(len(B_cos), len(B_sin))
        b_sin_norm = B_sin[:n_common] / np.sqrt(gap_sin[:n_common])
        b_cos_norm = B_cos[:n_common] / np.sqrt(gap_cos[:n_common])
        norm_s = np.linalg.norm(b_sin_norm)
        norm_c = np.linalg.norm(b_cos_norm)
        if norm_s > 1e-15 and norm_c > 1e-15:
            inner = np.dot(b_sin_norm / norm_s, b_cos_norm / norm_c)
        else:
            inner = 0.0

        # Gap ratios (j-wise comparison)
        gap_ratio = gap_sin[:n_common] / gap_cos[:n_common]

        row = {
            'lambda': int(lam), 'L': float(L),
            'D': float(D),
            'E_sin': float(E_sin), 'E_cos': float(E_cos),
            'diff': float(E_sin - E_cos), 'diff_over_D': float((E_sin - E_cos) / D),
            'sigma_sin': float(sigma_sin), 'sigma_cos': float(sigma_cos),
            'true_diff': float(sigma_sin - sigma_cos),
            'pt2_accuracy': float(pt2_accuracy),
            'cert_gap': float(cert_gap),
            'inner_product': float(inner),
            'gap_ratio_mean': float(np.mean(gap_ratio)),
            'gap_ratio_std': float(np.std(gap_ratio)),
        }
        results.append(row)

        print(f"\nlambda = {lam:5d}  (L = {L:.3f}, #primes = {len(primes)})")
        print(f"  D = {D:8.4f}  (Shift Parity diagonal advantage)")
        print(f"  E_sin = {E_sin:8.4f}, E_cos = {E_cos:8.4f}, diff = {E_sin-E_cos:+8.4f}")
        print(f"  diff/D = {(E_sin-E_cos)/D:.4f}  |  PT2 accuracy = {pt2_accuracy:.1f}%")
        print(f"  cert_gap = {cert_gap:+.4f}")
        print(f"  Common-mode inner product = {inner:.4f}")
        print(f"  Gap ratio sin/cos: mean = {np.mean(gap_ratio):.3f}, std = {np.std(gap_ratio):.3f}")
        sys.stdout.flush()

    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'lam':>6s} | {'D':>8s} | {'E_sin':>8s} | {'E_cos':>8s} | {'diff':>8s} | {'diff/D':>7s} | {'PT2%':>6s} | {'<b,b>':>6s} | {'cert_gap':>9s}")
    print("-" * 80)
    for r in results:
        print(f"{r['lambda']:6d} | {r['D']:8.3f} | {r['E_sin']:8.3f} | {r['E_cos']:8.3f} | {r['diff']:+8.3f} | {r['diff_over_D']:7.4f} | {r['pt2_accuracy']:6.1f} | {r['inner_product']:6.4f} | {r['cert_gap']:+9.4f}")

    # Scaling fits
    print("\n" + "=" * 80)
    print("SCALING FITS")
    print("=" * 80)

    lams = np.array([r['lambda'] for r in results])
    sqrtlams = np.sqrt(lams)
    Ls = np.array([r['L'] for r in results])
    Ds = np.array([r['D'] for r in results])
    diffs = np.array([r['diff'] for r in results])
    ratios = np.array([r['diff_over_D'] for r in results])

    # Fit D ~ a * sqrt(lam) / L
    D_norm = Ds * Ls / sqrtlams
    print(f"D * L / sqrt(lam): {D_norm[0]:.4f} ... {D_norm[-1]:.4f}  (should converge)")

    # Fit diff ~ a * lam^alpha
    mask = diffs > 0
    if np.sum(mask) > 3:
        c = np.polyfit(np.log(lams[mask]), np.log(diffs[mask]), 1)
        print(f"(E_sin - E_cos) ~ {np.exp(c[1]):.4f} * lam^{c[0]:.4f}")

    # Fit diff/D ~ a * L^alpha
    c2 = np.polyfit(np.log(Ls), np.log(ratios), 1)
    print(f"(E_sin - E_cos)/D ~ {np.exp(c2[1]):.4f} * L^{c2[0]:.4f}")

    # Fit diff/D ~ a / L
    print(f"\nTest diff/D * L = const?")
    for r in results:
        print(f"  lam={r['lambda']:5d}: diff/D * L = {r['diff_over_D'] * r['L']:.4f}")

    # Save results
    with open('scripts/_results/resolvent_analysis.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to scripts/_results/resolvent_analysis.json")

    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)
    print("1. diff/D ratio:", " -> ".join(f"{r['diff_over_D']:.3f}" for r in results))
    print("2. Common-mode inner product:", " -> ".join(f"{r['inner_product']:.3f}" for r in results))
    print("3. PT2 accuracy:", " -> ".join(f"{r['pt2_accuracy']:.0f}%" for r in results))
    print("=" * 80)
