#!/usr/bin/env python3
"""
weg2_even_odd_analysis.py
=========================
Analytische Untersuchung: WARUM dominiert der Even-Sektor fuer grosse lambda?

Hypothese: Der archimedische Kernel e^{u/2}/(2sinh(u)) hat eine Asymmetrie,
die bei grossem L = log(lambda) die geraden Moden bevorzugt.

Zerlege QW = W_diag + W_arch + W_prime
und berechne jeden Beitrag separat fuer Even vs Odd.

Ausserdem: Perron-Frobenius-Analyse -- ist der Kern positiv?
"""

import numpy as np
from scipy.linalg import eigh
from sympy import primerange
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

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


def build_QW_decomposed(lam, N, primes, basis='cos'):
    """Baue QW als Summe der drei Beitraege: diagonal + archimedisch + prim."""
    L = np.log(lam)
    n_quad = max(1000, 20 * N)
    n_int = max(600, 12 * N)

    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi = make_basis(N, t_grid, L, basis)

    # 1. Diagonal: (log4pi + gamma) * I
    W_diag = LOG4PI_GAMMA * np.eye(N)

    # 2. Archimedischer Kernel
    W_arch = np.zeros((N, N))
    s_max = min(2 * L, 10.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]

    for s in s_grid:
        k = np.exp(s / 2) / (2.0 * np.sinh(s))
        if k < 1e-15:
            continue
        Sp = (phi @ make_shifted(N, t_grid, L, s, basis).T) * dt
        Sm = (phi @ make_shifted(N, t_grid, L, -s, basis).T) * dt
        W_arch += (Sp + Sm - 2.0 * np.exp(-s / 2) * np.eye(N)) * k * ds

    # 3. Primzahl-Beitraege
    W_prime = np.zeros((N, N))
    for p in primes:
        logp = np.log(p)
        for m in range(1, 13):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for sign in [1.0, -1.0]:
                S = (phi @ make_shifted(N, t_grid, L, sign * shift, basis).T) * dt
                W_prime += coeff * S

    return W_diag, W_arch, W_prime


def analyze_decomposition():
    """Analysiere die drei Beitraege separat fuer Even vs Odd."""
    primes = list(primerange(2, 200))

    print("=" * 80)
    print("EVEN/ODD DEKOMPOSITION: Diag + Arch + Prime")
    print("=" * 80)

    N = 50
    lambdas = [10, 20, 25, 28, 30, 50, 100]

    print(f"\n{'lam':>5} | {'Sektor':>6} | {'l1_diag':>10} | {'l1_arch':>10} | "
          f"{'l1_prime':>10} | {'l1_total':>10} | {'l1_arch+p':>10}")
    print(f"{'-'*5}-+-{'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for lam in lambdas:
        primes_used = [p for p in primes if p <= max(lam, 100)]

        for basis in ['cos', 'sin']:
            sector = "EVEN" if basis == 'cos' else "ODD"
            W_d, W_a, W_p = build_QW_decomposed(lam, N, primes_used, basis)

            l1_d = np.sort(eigh(W_d, eigvals_only=True))[0]
            l1_a = np.sort(eigh(W_a, eigvals_only=True))[0]
            l1_p = np.sort(eigh(W_p, eigvals_only=True))[0]
            l1_total = np.sort(eigh(W_d + W_a + W_p, eigvals_only=True))[0]
            l1_ap = np.sort(eigh(W_a + W_p, eigvals_only=True))[0]

            print(f"{lam:5d} | {sector:>6} | {l1_d:+10.4f} | {l1_a:+10.4f} | "
                  f"{l1_p:+10.4f} | {l1_total:+10.4f} | {l1_ap:+10.4f}")
        print()


def analyze_matrix_structure():
    """Untersuche die Matrixstruktur: Sind Diagonale/Off-Diagonal verschieden?"""
    primes = list(primerange(2, 200))
    N = 30

    print("\n" + "=" * 80)
    print("MATRIX-STRUKTUR ANALYSE")
    print("=" * 80)

    for lam in [20, 30, 50, 100]:
        primes_used = [p for p in primes if p <= max(lam, 100)]
        print(f"\nlambda = {lam}:")

        for basis in ['cos', 'sin']:
            sector = "EVEN" if basis == 'cos' else "ODD"
            W_d, W_a, W_p = build_QW_decomposed(lam, N, primes_used, basis)
            W = W_d + W_a + W_p

            diag = np.diag(W)
            offdiag_norm = np.linalg.norm(W - np.diag(diag), 'fro')
            diag_range = (diag.min(), diag.max(), diag.mean())
            evals = np.sort(eigh(W, eigvals_only=True))

            print(f"  {sector}: diag=[{diag_range[0]:+.3f}, {diag_range[1]:+.3f}] "
                  f"mean={diag_range[2]:+.3f}, ||off||_F={offdiag_norm:.3f}, "
                  f"l1={evals[0]:+.4f}, l2={evals[1]:+.4f}")


def analyze_kernel_symmetry():
    """
    Der archimedische Kernel hat die Form:
    K(u) = e^{u/2}/(2sinh(u)) = 1/(2*(1-e^{-u}))  fuer u > 0

    Frage: Wie wirkt sich die Asymmetrie e^{u/2} auf gerade vs ungerade aus?

    Fuer gerade f: <Kf,f> = int K(u) [f*f~(u) + f*f~(-u)] du
    Fuer ungerade f: gleiches mit negativem Vorzeichen?

    Nein -- die QW-Form ist:
    QW(f,f) = diag + int_0^inf K(s) [S_s + S_{-s} - 2e^{-s/2}I] ds

    S_s(f)(t) = f(t-s) (Shift). Fuer Even: cos-Basis, fuer Odd: sin-Basis.
    Die Shift-Matrixelemente <phi_n, S_s phi_m> sind verschieden fuer cos vs sin.
    """
    print("\n" + "=" * 80)
    print("KERNEL-ASYMMETRIE ANALYSE")
    print("=" * 80)

    # Vergleiche Shift-Matrixelemente fuer cos vs sin
    N = 20
    for lam in [30, 100]:
        L = np.log(lam)
        n_quad = 2000
        t_grid = np.linspace(-L, L, n_quad)
        dt = t_grid[1] - t_grid[0]

        phi_cos = make_basis(N, t_grid, L, 'cos')
        phi_sin = make_basis(N, t_grid, L, 'sin')

        print(f"\nlambda={lam}, L={L:.3f}:")

        # Shift bei s = L/2 (halber Intervall)
        s_test = L / 2
        shifted_cos = make_shifted(N, t_grid, L, s_test, 'cos')
        shifted_sin = make_shifted(N, t_grid, L, s_test, 'sin')

        # Overlap-Matrizen
        O_cos = (phi_cos @ shifted_cos.T) * dt
        O_sin = (phi_sin @ shifted_sin.T) * dt

        # Trace = sum der Diagonalelemente = <phi_n, S_s phi_n>
        trace_cos = np.trace(O_cos)
        trace_sin = np.trace(O_sin)

        print(f"  Shift s=L/2={s_test:.3f}: trace_cos={trace_cos:+.6f}, "
              f"trace_sin={trace_sin:+.6f}, diff={trace_cos-trace_sin:+.6f}")

        # Der entscheidende Punkt: Die "Regularisierung" -2e^{-s/2}*I
        # zieht von ALLEN Diagonalelementen ab. Wenn trace_cos > trace_sin,
        # dann ist der effektive Beitrag fuer cos weniger negativ.
        reg = 2.0 * np.exp(-s_test / 2)
        eff_cos = trace_cos - N * reg
        eff_sin = trace_sin - N * reg
        print(f"  Regularisiert: eff_cos={eff_cos:+.6f}, eff_sin={eff_sin:+.6f}")

        # Integral ueber alle shifts mit Kernel
        print(f"  Integrierte Kernel-Beitraege (trace):")
        s_grid = np.linspace(0.01, min(2*L, 10), 500)
        ds = s_grid[1] - s_grid[0]
        total_cos = 0.0
        total_sin = 0.0
        for s in s_grid:
            k = np.exp(s / 2) / (2.0 * np.sinh(s))
            Sp_c = (phi_cos @ make_shifted(N, t_grid, L, s, 'cos').T) * dt
            Sm_c = (phi_cos @ make_shifted(N, t_grid, L, -s, 'cos').T) * dt
            Sp_s = (phi_sin @ make_shifted(N, t_grid, L, s, 'sin').T) * dt
            Sm_s = (phi_sin @ make_shifted(N, t_grid, L, -s, 'sin').T) * dt
            reg = 2.0 * np.exp(-s / 2)
            total_cos += k * (np.trace(Sp_c) + np.trace(Sm_c) - N * reg) * ds
            total_sin += k * (np.trace(Sp_s) + np.trace(Sm_s) - N * reg) * ds

        print(f"    Arch trace cos={total_cos:+.4f}, sin={total_sin:+.4f}, "
              f"diff={total_cos-total_sin:+.4f}")


if __name__ == "__main__":
    t0 = time.time()
    analyze_decomposition()
    analyze_matrix_structure()
    analyze_kernel_symmetry()
    print(f"\nTotal: {time.time()-t0:.1f}s")
