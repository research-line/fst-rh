#!/usr/bin/env python3
"""
weg2_server_cauchy_v2.py
========================
Server-Berechnung: Cauchy-Interlacing + Even/Odd-Vergleich
MIT KORREKTEM KERNEL: e^{u/2}/(2sinh(u))

Aenderungen gegenueber v1:
  - Korrekter archimedischer Kernel (Connes Gl. 10)
  - Even UND Odd Sektor
  - Mehr Primzahlen (bis 500)
  - Hoehere Quadratur-Aufloesung
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time
import json

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def make_basis(N, t_grid, L, basis_type='cos'):
    phi = np.zeros((N, len(t_grid)))
    if basis_type == 'cos':
        phi[0, :] = 1.0 / np.sqrt(2 * L)
        for n in range(1, N):
            phi[n, :] = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    else:
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


def build_QW(lam, N, primes, M_terms=12, n_quad=None, n_int=None, basis='cos'):
    """
    Galerkin-Matrix fuer QW_lambda mit KORREKTEM Kernel.
    Kernel: e^{u/2}/(2sinh(u))  [Connes Gl. 10, arXiv:2602.04022]
    """
    L = np.log(lam)
    if n_quad is None:
        n_quad = max(1000, 20 * N)
    if n_int is None:
        n_int = max(600, 12 * N)

    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = make_basis(N, t_grid, L, basis)

    W = LOG4PI_GAMMA * np.eye(N)

    s_max = min(2 * L, 10.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        # KORREKTER KERNEL: e^{s/2}/(2sinh(s))
        k = np.exp(s / 2) / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_shifted(N, t_grid, L, s, basis).T) * dt
        Sm = (phi @ make_shifted(N, t_grid, L, -s, basis).T) * dt
        W += (Sp + Sm - 2.0 * np.exp(-s / 2) * np.eye(N)) * k * ds

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


def cauchy_analysis(QW, N):
    """Cauchy-Interlacing Analyse fuer einen Sektor."""
    evals, evecs = eigh(QW)
    v0 = evecs[:, 0]
    gap = evals[1] - evals[0]

    n_star = np.argmax(np.abs(v0))
    weight_star = v0[n_star]**2

    top3 = np.argsort(np.abs(v0))[::-1][:3]
    top3_weights = v0[top3]**2

    # Cauchy bound: delete n_star
    mask = np.ones(N, dtype=bool)
    mask[n_star] = False
    QW_red = QW[np.ix_(mask, mask)]
    evals_red = np.sort(eigh(QW_red, eigvals_only=True))
    cauchy_bound = evals_red[0] - evals[0]

    return {
        'gap': float(gap),
        'lmin': float(evals[0]),
        'l2': float(evals[1]),
        'n_star': int(n_star),
        'weight_star': float(weight_star),
        'top3_modes': [int(x) for x in top3],
        'top3_weights': [float(x) for x in top3_weights],
        'cauchy_bound': float(cauchy_bound),
        'lmin_red': float(evals_red[0]),
    }


def compute_lambda(lam, primes, N_values):
    """Berechne Even und Odd Sektor fuer ein lambda."""
    results = []

    for N in N_values:
        t0 = time.time()

        QW_even = build_QW(lam, N, primes, basis='cos')
        even = cauchy_analysis(QW_even, N)
        even['sector'] = 'even'

        QW_odd = build_QW(lam, N, primes, basis='sin')
        odd_evals = np.sort(eigh(QW_odd, eigvals_only=True))

        elapsed = time.time() - t0

        # Even/Odd Vergleich
        dominant = "EVEN" if even['lmin'] < odd_evals[0] else "ODD"
        sector_gap = abs(even['lmin'] - odd_evals[0])

        r = {
            'N': N,
            'even': even,
            'odd_lmin': float(odd_evals[0]),
            'odd_l2': float(odd_evals[1]),
            'odd_gap': float(odd_evals[1] - odd_evals[0]),
            'dominant': dominant,
            'sector_gap': float(sector_gap),
            'elapsed': elapsed,
        }
        results.append(r)

        print(f"    N={N:3d}: even_l1={even['lmin']:+.6e} odd_l1={odd_evals[0]:+.6e} "
              f"=> {dominant:4s}  cauchy={even['cauchy_bound']:.4e} "
              f"even_gap={even['gap']:.4e}  ({elapsed:.1f}s)")

    return results


if __name__ == "__main__":
    print("=" * 80)
    print("SERVER: CAUCHY-INTERLACING v2 MIT KORREKTEM KERNEL e^{u/2}/(2sinh(u))")
    print("=" * 80)
    print(f"  LOG4PI_GAMMA = {LOG4PI_GAMMA:.8f}")

    all_primes = list(primerange(2, 500))
    print(f"  Primzahlen: {len(all_primes)} (bis {all_primes[-1]})")

    lambdas = [5, 8, 10, 13, 16, 20, 22, 25, 28, 30, 35, 40, 50,
               60, 80, 100, 130, 170, 200]

    all_results = {}

    for lam in lambdas:
        L = np.log(lam)
        primes_used = [p for p in all_primes if p <= max(lam, 100)]

        # N-Werte: starte hoeher, gehe weiter
        N_min = max(40, int(3 * L))
        N_max = max(100, int(8 * L))
        N_values = list(range(N_min, N_max + 1, 10))

        print(f"\n{'='*60}")
        print(f"  lambda={lam}, L={L:.3f}, 2L={2*L:.3f}, "
              f"primes={len(primes_used)}, N={N_values}")

        results = compute_lambda(lam, primes_used, N_values)
        final = results[-1]

        all_results[lam] = {
            'L': float(2 * L),
            'n_primes': len(primes_used),
            'N_final': final['N'],
            'even_lmin': final['even']['lmin'],
            'even_gap': final['even']['gap'],
            'even_cauchy': final['even']['cauchy_bound'],
            'even_n_star': final['even']['n_star'],
            'even_weight': final['even']['weight_star'],
            'odd_lmin': final['odd_lmin'],
            'odd_gap': final['odd_gap'],
            'dominant': final['dominant'],
            'sector_gap': final['sector_gap'],
            'convergence': results,
        }

    # === ZUSAMMENFASSUNG ===
    print(f"\n\n{'='*80}")
    print("ZUSAMMENFASSUNG: CAUCHY-INTERLACING v2 (KORREKTER KERNEL)")
    print(f"{'='*80}")

    print(f"\n  EVEN-SEKTOR GAPS:")
    print(f"  {'lam':>5} | {'N':>3} | {'l1_even':>12} | {'gap':>12} | "
          f"{'cauchy':>12} | {'n*':>4} | {'|v0|^2':>8}")
    print(f"  {'-'*5}-+-{'-'*3}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*4}-+-{'-'*8}")

    for lam in lambdas:
        r = all_results[lam]
        print(f"  {lam:5d} | {r['N_final']:3d} | {r['even_lmin']:+12.6e} | "
              f"{r['even_gap']:12.6e} | {r['even_cauchy']:+12.6e} | "
              f"{r['even_n_star']:4d} | {r['even_weight']:8.4f}")

    print(f"\n  EVEN/ODD VERGLEICH:")
    print(f"  {'lam':>5} | {'l1_even':>12} | {'l1_odd':>12} | {'Sektor':>6} | "
          f"{'Sek.Gap':>10} | {'Thm 6.1':>8}")
    print(f"  {'-'*5}-+-{'-'*12}-+-{'-'*12}-+-{'-'*6}-+-{'-'*10}-+-{'-'*8}")

    n_even = 0
    n_thm61 = 0
    for lam in lambdas:
        r = all_results[lam]
        is_even = r['dominant'] == 'EVEN'
        thm61 = is_even and r['even_cauchy'] > 0
        if is_even:
            n_even += 1
        if thm61:
            n_thm61 += 1
        print(f"  {lam:5d} | {r['even_lmin']:+12.6e} | {r['odd_lmin']:+12.6e} | "
              f"{r['dominant']:>6} | {r['sector_gap']:10.4e} | "
              f"{'JA' if thm61 else 'NEIN':>8}")

    print(f"\n  EVEN-Grundzustand: {n_even}/{len(lambdas)}")
    print(f"  Theorem 6.1 anwendbar: {n_thm61}/{len(lambdas)}")

    all_cauchy_pos = all(all_results[l]['even_cauchy'] > 0 for l in lambdas)
    min_cauchy = min(all_results[l]['even_cauchy'] for l in lambdas)
    print(f"  Alle Even-Cauchy-Schranken > 0: {'JA' if all_cauchy_pos else 'NEIN'}")
    print(f"  Minimale Even-Cauchy-Schranke: {min_cauchy:.6e}")

    # Speichere
    outfile = "cauchy_v2_results_server.json"
    with open(outfile, 'w') as f:
        # Nur die Zusammenfassungsdaten, nicht die vollen Konvergenzdaten
        summary = {}
        for lam in lambdas:
            r = all_results[lam]
            summary[str(lam)] = {k: v for k, v in r.items() if k != 'convergence'}
        json.dump(summary, f, indent=2)
    print(f"\n  Ergebnisse gespeichert: {outfile}")
