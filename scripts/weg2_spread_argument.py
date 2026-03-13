#!/usr/bin/env python3
"""
weg2_spread_argument.py
========================
Eigenwertspreizungs-Argument fuer Even-Dominanz.

KERNIDEE:
Fuer eine symmetrische NxN Matrix M mit Eigenwerten l1 <= l2 <= ... <= lN:
  Tr(M) = sum l_i
  ||M||_F^2 = sum l_i^2

Die Varianz der Eigenwerte ist:
  Var = (1/N) sum l_i^2 - (Tr/N)^2 = (||M||_F^2 - Tr^2/N) / N

Groessere Varianz => groessere Spreizung => l1 weiter unter dem Mittelwert.

ARGUMENT:
1. QW = C*I + W_arch + W_prime
2. W_arch ist paritaetsneutral => gleiche Varianz fuer cos/sin
3. W_prime hat GROESSERE ||.||_F im cos-Sektor (weil cos-Matrixelemente groesser sind)
4. Groessere ||W_prime||_F => groessere Gesamtvarianz => l1(cos) < l1(sin)

FORMALISIERUNG:
l1 >= Tr/N - sqrt((N-1)/N * (||M||_F^2 - Tr^2/N))  (Schur-Horn)
l1 <= Tr/N - sqrt(1/(N-1) * (||M||_F^2 - Tr^2/N))  (Gegenseite, falls l1 isoliert)

Die OBERE Schranke fuer l1 haengt von der Off-Diagonal-Norm ab:
||off(M)||_F^2 = ||M||_F^2 - ||diag(M)||^2 = sum_{i!=j} M_{ij}^2
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import sys
sys.path.insert(0, r'C:\Users\User\OneDrive\.RESEARCH\Natur&Technik\1 Musterbeweise\RH\scripts')

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def spread_analysis():
    """Vergleiche die Eigenwertspreizung von W_cos vs W_sin."""
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))
    N = 30

    print("=" * 80)
    print("EIGENWERTSPREIZUNGS-ARGUMENT")
    print("=" * 80)

    print(f"\n{'lam':>5} | {'Tr_c':>8} | {'Tr_s':>8} | {'||F||_c':>8} | {'||F||_s':>8} | "
          f"{'||off||_c':>8} | {'||off||_s':>8} | {'l1_c':>8} | {'l1_s':>8} | {'Delta':>8}")
    print("-" * 100)

    for lam in [30, 40, 50, 70, 100, 150, 200, 300, 500]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        tr_c = np.trace(W_cos)
        tr_s = np.trace(W_sin)
        fro_c = np.linalg.norm(W_cos, 'fro')
        fro_s = np.linalg.norm(W_sin, 'fro')
        off_c = np.sqrt(fro_c**2 - np.sum(np.diag(W_cos)**2))
        off_s = np.sqrt(fro_s**2 - np.sum(np.diag(W_sin)**2))

        evals_c = np.sort(eigh(W_cos, eigvals_only=True))
        evals_s = np.sort(eigh(W_sin, eigvals_only=True))

        print(f"{lam:5d} | {tr_c:+8.2f} | {tr_s:+8.2f} | {fro_c:8.2f} | {fro_s:8.2f} | "
              f"{off_c:8.2f} | {off_s:8.2f} | {evals_c[0]:+8.3f} | {evals_s[0]:+8.3f} | "
              f"{evals_c[0]-evals_s[0]:+8.3f}")


def schur_horn_bounds():
    """
    Schur-Horn Schranken fuer l1.

    Untere Schranke: l1 >= Tr/N - sqrt((N-1) * Var)
    wobei Var = (||M||_F^2/N - (Tr/N)^2)

    Das ist l1 >= mean - sqrt((N-1) * Var).

    Effektive Schranke:
    l1 <= mean - ||off(M)||_F / sqrt(N-1)

    Wenn ||off_cos|| > ||off_sin||, dann ist die OBERE Schranke fuer l1_cos
    niedriger als fuer l1_sin.
    """
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))
    N = 30

    print("\n" + "=" * 80)
    print("SCHUR-HORN SCHRANKEN")
    print("=" * 80)

    print(f"\n{'lam':>5} | {'mean_c':>8} | {'mean_s':>8} | {'off_c/sqrt':>10} | "
          f"{'off_s/sqrt':>10} | {'bound_c':>8} | {'bound_s':>8} | {'l1_c':>8} | {'l1_s':>8}")
    print("-" * 100)

    for lam in [30, 50, 100, 200, 500]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        mean_c = np.trace(W_cos) / N
        mean_s = np.trace(W_sin) / N

        off_c = np.sqrt(np.linalg.norm(W_cos, 'fro')**2 - np.sum(np.diag(W_cos)**2))
        off_s = np.sqrt(np.linalg.norm(W_sin, 'fro')**2 - np.sum(np.diag(W_sin)**2))

        # Obere Schranke fuer l1: mean - off/sqrt(N-1)
        bound_c = mean_c - off_c / np.sqrt(N - 1)
        bound_s = mean_s - off_s / np.sqrt(N - 1)

        evals_c = np.sort(eigh(W_cos, eigvals_only=True))
        evals_s = np.sort(eigh(W_sin, eigvals_only=True))

        print(f"{lam:5d} | {mean_c:+8.3f} | {mean_s:+8.3f} | {off_c/np.sqrt(N-1):10.3f} | "
              f"{off_s/np.sqrt(N-1):10.3f} | {bound_c:+8.3f} | {bound_s:+8.3f} | "
              f"{evals_c[0]:+8.3f} | {evals_s[0]:+8.3f}")


def off_diagonal_decomposition():
    """
    Zerlege ||off||_F^2 in Arch- und Prim-Beitraege.

    ||off(W)||^2 = ||off(W_arch)||^2 + ||off(W_prime)||^2 + 2*<off(W_arch), off(W_prime)>

    Wenn W_arch paritaetsneutral, ist ||off(W_arch_cos)|| ~ ||off(W_arch_sin)||.
    Die DIFFERENZ kommt von W_prime:
    ||off(W_cos)|| - ||off(W_sin)|| ~ ||off(W_prime_cos)|| - ||off(W_prime_sin)||
    """
    from sympy import primerange
    from weg2_analytic_even_odd import shift_element_cos, shift_element_sin

    primes = list(primerange(2, 200))
    N = 25

    print("\n" + "=" * 80)
    print("OFF-DIAGONAL ZERLEGUNG: Arch vs Prime")
    print("=" * 80)

    for lam in [30, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)

        # Baue W_arch separat fuer cos und sin
        W_arch_cos = np.zeros((N, N))
        W_arch_sin = np.zeros((N, N))
        n_int = max(800, 15 * N)
        s_max = min(2 * L, 10.0)
        s_grid = np.linspace(0.005, s_max, n_int)
        ds = s_grid[1] - s_grid[0]

        for s in s_grid:
            k = np.exp(s / 2) / (2.0 * np.sinh(s))
            if k < 1e-15:
                continue
            for i in range(N):
                for j in range(i, N):
                    sp_c = shift_element_cos(i, j, s, L)
                    sm_c = shift_element_cos(i, j, -s, L)
                    sp_s = shift_element_sin(i, j, s, L)
                    sm_s = shift_element_sin(i, j, -s, L)
                    reg = -2.0 * np.exp(-s / 2) * (1.0 if i == j else 0.0)
                    val_c = k * (sp_c + sm_c + reg) * ds
                    val_s = k * (sp_s + sm_s + reg) * ds
                    W_arch_cos[i, j] += val_c
                    W_arch_sin[i, j] += val_s
                    if i != j:
                        W_arch_cos[j, i] += val_c
                        W_arch_sin[j, i] += val_s

        # Baue W_prime separat
        W_prime_cos = np.zeros((N, N))
        W_prime_sin = np.zeros((N, N))
        for p in primes_used:
            logp = np.log(p)
            for m_exp in range(1, 20):
                coeff = logp * p**(-m_exp / 2.0)
                shift = m_exp * logp
                if shift >= 2 * L or coeff < 1e-15:
                    break
                for i in range(N):
                    for j in range(i, N):
                        sc = shift_element_cos(i, j, shift, L) + shift_element_cos(i, j, -shift, L)
                        ss = shift_element_sin(i, j, shift, L) + shift_element_sin(i, j, -shift, L)
                        W_prime_cos[i, j] += coeff * sc
                        W_prime_sin[i, j] += coeff * ss
                        if i != j:
                            W_prime_cos[j, i] += coeff * sc
                            W_prime_sin[j, i] += coeff * ss

        off_arch_c = np.sqrt(np.linalg.norm(W_arch_cos, 'fro')**2 - np.sum(np.diag(W_arch_cos)**2))
        off_arch_s = np.sqrt(np.linalg.norm(W_arch_sin, 'fro')**2 - np.sum(np.diag(W_arch_sin)**2))
        off_prime_c = np.sqrt(np.linalg.norm(W_prime_cos, 'fro')**2 - np.sum(np.diag(W_prime_cos)**2))
        off_prime_s = np.sqrt(np.linalg.norm(W_prime_sin, 'fro')**2 - np.sum(np.diag(W_prime_sin)**2))

        print(f"\nlambda={lam}:")
        print(f"  ||off(W_arch_cos)||  = {off_arch_c:.4f}")
        print(f"  ||off(W_arch_sin)||  = {off_arch_s:.4f}")
        print(f"  Ratio arch:          = {off_arch_c/off_arch_s:.4f}")
        print(f"  ||off(W_prime_cos)|| = {off_prime_c:.4f}")
        print(f"  ||off(W_prime_sin)|| = {off_prime_s:.4f}")
        print(f"  Ratio prime:         = {off_prime_c/off_prime_s:.4f}")

        # Gesamt-Off-Diagonal
        W_cos = LOG4PI_GAMMA * np.eye(N) + W_arch_cos + W_prime_cos
        W_sin = LOG4PI_GAMMA * np.eye(N) + W_arch_sin + W_prime_sin
        off_tot_c = np.sqrt(np.linalg.norm(W_cos, 'fro')**2 - np.sum(np.diag(W_cos)**2))
        off_tot_s = np.sqrt(np.linalg.norm(W_sin, 'fro')**2 - np.sum(np.diag(W_sin)**2))
        print(f"  ||off(W_total_cos)|| = {off_tot_c:.4f}")
        print(f"  ||off(W_total_sin)|| = {off_tot_s:.4f}")
        print(f"  Ratio total:         = {off_tot_c/off_tot_s:.4f}")


def formal_spread_theorem():
    """
    SATZ-VERSUCH:

    Sei W = A + B mit A paritaetsneutral (||off(A_cos)|| = ||off(A_sin)||)
    und B mit ||off(B_cos)|| > ||off(B_sin)||.
    Sei Tr(W_cos) >= Tr(W_sin) (oder auch nicht -- pruefen!).

    Dann: l1(W_cos) < l1(W_sin) WENN:
    ||off(W_cos)||/sqrt(N-1) - Tr(W_cos)/N > ||off(W_sin)||/sqrt(N-1) - Tr(W_sin)/N

    D.h.:  [||off(W_cos)|| - ||off(W_sin)||] / sqrt(N-1) > [Tr(W_cos) - Tr(W_sin)] / N

    Das heisst: Die OFF-DIAGONAL-DIFFERENZ muss die TRACE-DIFFERENZ ueberwiegen.
    """
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))
    N = 30

    print("\n" + "=" * 80)
    print("FORMALER SPREIZUNGS-TEST")
    print("=" * 80)

    print(f"\n  Bedingung: [||off_c||-||off_s||]/sqrt(N-1) > [Tr_c-Tr_s]/N")

    for lam in [30, 50, 100, 200, 500]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        tr_diff = np.trace(W_cos) - np.trace(W_sin)
        off_c = np.sqrt(np.linalg.norm(W_cos, 'fro')**2 - np.sum(np.diag(W_cos)**2))
        off_s = np.sqrt(np.linalg.norm(W_sin, 'fro')**2 - np.sum(np.diag(W_sin)**2))
        off_diff = off_c - off_s

        lhs = off_diff / np.sqrt(N - 1)
        rhs = tr_diff / N

        evals_c = np.sort(eigh(W_cos, eigvals_only=True))
        evals_s = np.sort(eigh(W_sin, eigvals_only=True))

        satisfied = lhs > rhs

        print(f"\n  lambda={lam}:")
        print(f"    LHS = {lhs:+.4f}, RHS = {rhs:+.4f} => {'ERFUELLT' if satisfied else 'NICHT'}")
        print(f"    ||off_c||={off_c:.2f}, ||off_s||={off_s:.2f}, diff={off_diff:+.2f}")
        print(f"    Tr_c={np.trace(W_cos):+.2f}, Tr_s={np.trace(W_sin):+.2f}, diff={tr_diff:+.2f}")
        print(f"    l1_c={evals_c[0]:+.4f}, l1_s={evals_s[0]:+.4f}, Delta={evals_c[0]-evals_s[0]:+.4f}")

        # Schaerfere Schranke: Verwende die tatsaechliche EW-Verteilung
        # l1 <= mean - sqrt(sum_{i>1} (li - mean)^2 / (N-1))
        # = mean - sqrt((||M||_F^2/N - mean^2) * N / (N-1))  ... nein

        # Eigentlich: l1 = mean - sigma * z1, wobei z1 ~ sqrt(N) fuer Wigner-artige Matrizen
        mean_c = np.trace(W_cos) / N
        mean_s = np.trace(W_sin) / N
        var_c = np.linalg.norm(W_cos, 'fro')**2 / N - mean_c**2
        var_s = np.linalg.norm(W_sin, 'fro')**2 / N - mean_s**2
        print(f"    Var_c={var_c:.2f}, Var_s={var_s:.2f}")
        print(f"    l1_c - mean_c = {evals_c[0]-mean_c:+.4f}, l1_s - mean_s = {evals_s[0]-mean_s:+.4f}")
        print(f"    Relative Abweichung: c: {(evals_c[0]-mean_c)/np.sqrt(var_c):.3f}, "
              f"s: {(evals_s[0]-mean_s)/np.sqrt(var_s):.3f} (in Std-Abw.)")


if __name__ == "__main__":
    import time
    t0 = time.time()
    spread_analysis()
    schur_horn_bounds()
    off_diagonal_decomposition()
    formal_spread_theorem()
    print(f"\nTotal: {time.time()-t0:.1f}s")
