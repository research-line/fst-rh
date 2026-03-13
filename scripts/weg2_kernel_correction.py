#!/usr/bin/env python3
"""
weg2_kernel_correction.py
=========================
KERNEL-VERIFIZIERUNG (BEWEISNOTIZ Item 18):

Connes Gl. 10 (arXiv:2602.04022v1) definiert:
  W_R(f) = (log 4pi + gamma) f(1)
           + int_1^inf [f(x) + f(x^-1) - 2x^{-1/2}f(1)] * x^{1/2}/(x-x^{-1}) d*x

In additiven Koordinaten (u = log x, d*x = du):
  Kernel = e^{u/2} / (2 sinh(u))

NICHT 1/(2 sinh(u)) wie bisher im Code!

Fuer die quadratische Form QW(f,f) = W_R(f*f~):
  - h(0) = ||f||^2 => Identity (korrekt)
  - Der Kernel-Faktor e^{u/2}/(2sinh(u)) bleibt unveraendert
  - Die Subtraktion -2e^{-u/2} bleibt auch (sie ist VOR dem Kernel)
  - Primterme haben keinen Kernel-Faktor (korrekt)

FIX: k = exp(s/2) / (2*sinh(s))  statt  k = 1 / (2*sinh(s))
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def make_cos_basis(N, t_grid, L):
    phi = np.zeros((N, len(t_grid)))
    phi[0, :] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N):
        phi[n, :] = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi


def make_sin_basis(N, t_grid, L):
    phi = np.zeros((N, len(t_grid)))
    for n in range(N):
        phi[n, :] = np.sin((n + 1) * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi


def make_shifted(N, t_grid, L, shift, basis_type='cos'):
    ts = t_grid - shift
    mask = np.abs(ts) <= L
    phi = np.zeros((N, len(t_grid)))
    if basis_type == 'cos':
        phi[0, mask] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, mask] = np.cos(n * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    else:
        for n in range(N):
            phi[n, mask] = np.sin((n + 1) * np.pi * ts[mask] / (2 * L)) / np.sqrt(L)
    return phi


def build_QW(lam, N, primes, M_terms=12, n_quad=800, n_int=500,
             basis='cos', use_correct_kernel=True):
    """
    Galerkin-Matrix fuer QW_lambda.

    use_correct_kernel=True:  k = exp(s/2) / (2 sinh(s))  [KORREKT]
    use_correct_kernel=False: k = 1 / (2 sinh(s))          [ALTER CODE]
    """
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]

    if basis == 'cos':
        phi = make_cos_basis(N, t_grid, L)
    else:
        phi = make_sin_basis(N, t_grid, L)

    # Diagonalterm: (log 4pi + gamma) * I  (korrekt fuer quadratische Form)
    W = LOG4PI_GAMMA * np.eye(N)

    # Archimedischer Integral-Term
    s_max = min(2 * L, 8.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        if use_correct_kernel:
            k = np.exp(s / 2) / (2.0 * np.sinh(s))
        else:
            k = 1.0 / (2.0 * np.sinh(s))

        if k < 1e-15:
            continue

        Sp = (phi @ make_shifted(N, t_grid, L, s, basis).T) * dt
        Sm = (phi @ make_shifted(N, t_grid, L, -s, basis).T) * dt
        W += (Sp + Sm - 2.0 * np.exp(-s / 2) * np.eye(N)) * k * ds

    # Primzahl-Terme (kein Kernel-Faktor)
    for p in primes:
        logp = np.log(p)
        for m in range(1, M_terms + 1):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_shifted(N, t_grid, L, sign * shift, basis).T) * dt
                W += coeff * S

    return W


def run_kernel_comparison(lambdas, primes, N=50):
    """Vergleiche alten und korrigierten Kernel."""
    print("=" * 80)
    print("KERNEL-VERIFIZIERUNG: e^{u/2}/(2sinh(u)) vs 1/(2sinh(u))")
    print("=" * 80)
    print(f"\nParameter: N_basis={N}, n_quad=800, n_int=500")
    print(f"Primzahlen: {len(primes)} (bis {primes[-1]})")

    results = []

    for lam in lambdas:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 47)]
        N_used = max(N, int(3 * L))

        t0 = time.time()

        # KORREKTER Kernel
        QW_even_new = build_QW(lam, N_used, primes_used, basis='cos', use_correct_kernel=True)
        QW_odd_new = build_QW(lam, N_used, primes_used, basis='sin', use_correct_kernel=True)
        ev_even_new = np.sort(eigh(QW_even_new, eigvals_only=True))
        ev_odd_new = np.sort(eigh(QW_odd_new, eigvals_only=True))

        # ALTER Kernel
        QW_even_old = build_QW(lam, N_used, primes_used, basis='cos', use_correct_kernel=False)
        QW_odd_old = build_QW(lam, N_used, primes_used, basis='sin', use_correct_kernel=False)
        ev_even_old = np.sort(eigh(QW_even_old, eigvals_only=True))
        ev_odd_old = np.sort(eigh(QW_odd_old, eigvals_only=True))

        elapsed = time.time() - t0

        # Analyse
        sector_new = "EVEN" if ev_even_new[0] < ev_odd_new[0] else "ODD"
        sector_old = "EVEN" if ev_even_old[0] < ev_odd_old[0] else "ODD"

        gap_even_new = ev_even_new[1] - ev_even_new[0]
        gap_even_old = ev_even_old[1] - ev_even_old[0]

        r = {
            'lam': lam, 'N': N_used,
            'l1_even_new': ev_even_new[0], 'l1_odd_new': ev_odd_new[0],
            'l1_even_old': ev_even_old[0], 'l1_odd_old': ev_odd_old[0],
            'gap_even_new': gap_even_new, 'gap_even_old': gap_even_old,
            'sector_new': sector_new, 'sector_old': sector_old,
            'sector_gap_new': abs(ev_even_new[0] - ev_odd_new[0]),
            'sector_gap_old': abs(ev_even_old[0] - ev_odd_old[0]),
        }
        results.append(r)

        print(f"\n  lambda={lam:4d} (N={N_used}, {elapsed:.1f}s):")
        print(f"    KORREKT:  l1_even={ev_even_new[0]:+.6e}, l1_odd={ev_odd_new[0]:+.6e} "
              f"=> {sector_new}  gap_even={gap_even_new:.4e}")
        print(f"    ALTER:    l1_even={ev_even_old[0]:+.6e}, l1_odd={ev_odd_old[0]:+.6e} "
              f"=> {sector_old}  gap_even={gap_even_old:.4e}")
        print(f"    SHIFT:    even={ev_even_new[0]-ev_even_old[0]:+.4e}, "
              f"odd={ev_odd_new[0]-ev_odd_old[0]:+.4e}")

    # Zusammenfassung
    print(f"\n{'='*80}")
    print("ZUSAMMENFASSUNG")
    print(f"{'='*80}")
    print(f"\n  {'lam':>5} | {'l1_even (NEU)':>14} | {'l1_odd (NEU)':>14} | "
          f"{'Sektor':>6} | {'Gap Even':>10} | {'Sektor ALT':>10}")
    print(f"  {'-'*5}-+-{'-'*14}-+-{'-'*14}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}")

    n_even_new = 0
    n_even_old = 0
    for r in results:
        if r['sector_new'] == 'EVEN':
            n_even_new += 1
        if r['sector_old'] == 'EVEN':
            n_even_old += 1
        print(f"  {r['lam']:5d} | {r['l1_even_new']:+14.6e} | {r['l1_odd_new']:+14.6e} | "
              f"{r['sector_new']:>6} | {r['gap_even_new']:10.4e} | {r['sector_old']:>10}")

    print(f"\n  EVEN-Grundzustand (KORREKT): {n_even_new}/{len(results)}")
    print(f"  EVEN-Grundzustand (ALTER):   {n_even_old}/{len(results)}")

    # Kernel-Faktor Analyse
    print(f"\n{'='*80}")
    print("KERNEL-FAKTOR ANALYSE")
    print(f"{'='*80}")
    print("\n  u       | 1/(2sinh(u)) | e^{u/2}/(2sinh(u)) | Ratio")
    print(f"  {'-'*8}-+-{'-'*13}-+-{'-'*19}-+-{'-'*8}")
    for u in [0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]:
        k_old = 1.0 / (2 * np.sinh(u))
        k_new = np.exp(u / 2) / (2 * np.sinh(u))
        print(f"  {u:8.2f} | {k_old:13.6e} | {k_new:19.6e} | {k_new/k_old:8.4f}")

    return results


if __name__ == "__main__":
    primes = list(primerange(2, 200))
    lambdas = [5, 8, 10, 13, 16, 20, 25, 30, 40, 50, 80, 100, 200]

    results = run_kernel_comparison(lambdas, primes, N=50)
