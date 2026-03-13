#!/usr/bin/env python3
"""
weg2_total_positivity_test.py
==============================
Teste die Totale Positivitaet des archimedischen Kerns.

Ein Kern K(s) ist total positiv (TP), wenn fuer ALLE n und alle
  0 < s_1 < s_2 < ... < s_n, 0 < t_1 < t_2 < ... < t_n
die Determinante det[K(s_i, t_j)]_{i,j=1}^n >= 0 ist.

Fuer einen Faltungskern K(s,t) = k(s-t) ist dies aequivalent zu:
  k ist Polya-Frequenz-Funktion (PF_infinity).

Fuer unseren Kern: k(u) = e^{u/2}/(2*sinh(u)) = 1/(2*(1 - e^{-u})) fuer u > 0.

AUCH: Der VOLLE Weil-Kern (mit Primtermen) ist NICHT TP (Primterme sind Deltas
mit wechselnden Positionen). Aber der archimedische Anteil koennte TP sein.

Ausserdem: Teste den VOLLEN Kern als Matrix (Galerkin-diskretisiert).

SCHOENBERG'S THEOREM: k ist PF_infinity <=> 1/k-hat(z) hat keine Nullstellen
in Im(z) >= 0. Aequivalent: k-hat(z) = exp(a*z^2 + b*z) * prod (1 + c_j*z) * exp(-c_j*z)
mit a >= 0, b reell, c_j reell.
"""

import numpy as np
from scipy.linalg import det
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def kernel_arch(u):
    """Archimedischer Kern e^{u/2}/(2*sinh(u)) fuer u > 0."""
    if abs(u) < 1e-12:
        return 0.5  # Grenzwert: lim_{u->0} u/(2*sinh(u)) * e^{u/2}/u -> 1/2
    return np.exp(u / 2) / (2 * np.sinh(u))


def test_TP_minors(kernel_func, s_values, order_max=6):
    """
    Teste TP: Berechne ALLE Minoren bis Ordnung order_max.

    Fuer einen Faltungskern K(s_i, t_j) = k(s_i - t_j) testen wir:
    det[k(s_i - t_j)] >= 0 fuer s_1 < s_2 < ... < s_n
    mit t_j = s_j (Diagonaltest) und mit zufaelligen t-Werten.
    """
    n = len(s_values)
    results = {}

    for order in range(1, min(order_max + 1, n + 1)):
        min_det = float('inf')
        max_det = float('-inf')
        n_neg = 0
        n_tests = 0

        # Systematisch: alle n-ueber-order Kombinationen von s und t
        from itertools import combinations
        s_combos = list(combinations(range(n), order))

        for s_idx in s_combos:
            for t_idx in s_combos:
                M = np.zeros((order, order))
                for i, si in enumerate(s_idx):
                    for j, tj in enumerate(t_idx):
                        M[i, j] = kernel_func(s_values[si] - s_values[tj])
                d = det(M)
                min_det = min(min_det, d)
                max_det = max(max_det, d)
                if d < -1e-10:
                    n_neg += 1
                n_tests += 1

        results[order] = {
            'min_det': min_det,
            'max_det': max_det,
            'n_neg': n_neg,
            'n_tests': n_tests,
        }

    return results


def test_arch_kernel_TP():
    """Teste TP des archimedischen Kerns."""
    print("=" * 80)
    print("TOTALE POSITIVITAET: Archimedischer Kern e^{u/2}/(2*sinh(u))")
    print("=" * 80)

    # Teste mit verschiedenen Gitterpunkten
    for n_pts in [8, 12, 16]:
        for s_max in [2.0, 5.0, 10.0]:
            s_values = np.linspace(0.05, s_max, n_pts)
            print(f"\n  n={n_pts}, s in [0.05, {s_max}]:")

            results = test_TP_minors(kernel_arch, s_values, order_max=5)

            for order, r in results.items():
                status = "TP" if r['n_neg'] == 0 else f"NICHT TP ({r['n_neg']}/{r['n_tests']} neg)"
                print(f"    Ordnung {order}: min={r['min_det']:+.6e}, "
                      f"max={r['max_det']:+.6e} -> {status}")


def test_shifted_kernel_TP():
    """
    Der archimedische OPERATOR hat die Form:
    (Kf)(t) = int_0^{s_max} K(s) * [f(t-s) + f(t+s) - 2*e^{-s/2}*f(t)] ds

    Das ist KEIN einfacher Faltungsoperator! Die Subtraktion -2*e^{-s/2}*f(t)
    zerstoert die TP-Eigenschaft.

    Aber: Betrachte den POSITIVEN Teil K(s)*[f(t-s) + f(t+s)] separat.
    Und den "Regularierer" -2*K(s)*e^{-s/2}*f(t) = -2*e^0/(2*sinh(s))*f(t)
    = -1/sinh(s)*f(t) separat. Der Regularierer ist ein Rang-1-Operator!
    """
    print("\n" + "=" * 80)
    print("ZERLEGUNG: K_arch = K_shift + K_reg (Rang-1)")
    print("=" * 80)

    # K_shift(s) = e^{s/2}/(2*sinh(s)) = positiver Faltungskern (TP-Kandidat)
    # K_reg(s) = -2*e^{-s/2} * K(s) = -1/sinh(s) (negativ! -> zerstoert TP)

    print("\n  K_shift(s) = e^{s/2}/(2*sinh(s)): TP-Test auf positivem Halbraum")
    s_values = np.linspace(0.1, 8.0, 12)
    results = test_TP_minors(kernel_arch, s_values, order_max=5)
    for order, r in results.items():
        status = "TP" if r['n_neg'] == 0 else f"NICHT TP ({r['n_neg']})"
        print(f"    Ordnung {order}: min={r['min_det']:+.6e} -> {status}")

    # Teste den symmetrisierten Kern K(|s-t|)
    print("\n  Symmetrisierter Kern K(|s_i - t_j|):")

    def kernel_sym(u):
        return kernel_arch(abs(u))

    results = test_TP_minors(kernel_sym, s_values, order_max=5)
    for order, r in results.items():
        status = "TP" if r['n_neg'] == 0 else f"NICHT TP ({r['n_neg']})"
        print(f"    Ordnung {order}: min={r['min_det']:+.6e} -> {status}")


def test_full_QW_positivity():
    """
    Teste Positivitaets-Eigenschaften der VOLLEN QW-Matrix.

    Frage: Hat die QW-Matrix (Even oder Odd) spezielle Positivitaets-Struktur?
    - Ist Q + c*I total positiv fuer ein c?
    - Hat Q eine Faktorisierung Q = L*D*L^T mit spezieller Struktur?
    """
    from sympy import primerange
    from scipy.linalg import eigh

    primes = list(primerange(2, 100))
    print("\n" + "=" * 80)
    print("QW-MATRIX: POSITIVITAETS-ANALYSE")
    print("=" * 80)

    # Importiere aus bestehendem Code
    import sys
    sys.path.insert(0, '.')
    from weg2_analytic_even_odd import build_QW_analytic

    N = 15  # Klein genug fuer Minor-Berechnung

    for lam in [30, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        print(f"\nlambda={lam}, N={N}:")

        for basis in ['cos', 'sin']:
            sector = "EVEN" if basis == 'cos' else "ODD"
            W = build_QW_analytic(lam, N, primes_used, basis)
            evals = np.sort(eigh(W, eigvals_only=True))

            # Shift: Q' = Q - l1*I (so dass l1' = 0)
            shift = evals[0]
            W_shifted = W - shift * np.eye(N)

            # Teste: Ist W_shifted TP?
            # (Alle Eigenwerte >= 0 -> PSD -> alle Hauptminoren >= 0)
            # Aber TP erfordert ALLE Minoren >= 0, nicht nur Hauptminoren

            # Teste alle 2x2 Minoren
            n_neg_2 = 0
            min_minor_2 = float('inf')
            for i in range(N):
                for j in range(i + 1, N):
                    for k in range(N):
                        for l in range(k + 1, N):
                            m = W_shifted[i, k] * W_shifted[j, l] - W_shifted[i, l] * W_shifted[j, k]
                            min_minor_2 = min(min_minor_2, m)
                            if m < -1e-10:
                                n_neg_2 += 1

            total_2 = N * (N - 1) // 2
            total_2 = total_2 * total_2
            print(f"  {sector}: l1={evals[0]:+.4f}, shifted 2x2-Minoren: "
                  f"min={min_minor_2:+.6e}, neg={n_neg_2}/{total_2}")

            # Teste leading principal minors (notwendig fuer PD)
            lpm = []
            for k in range(1, min(8, N + 1)):
                lpm.append(det(W_shifted[:k, :k]))
            print(f"    Leading principal minors: {[f'{x:.4f}' for x in lpm]}")


def fourier_analysis():
    """
    Schoenberg's Theorem: k ist PF_infinity <=> 1/k-hat hat keine Nullstellen
    in oberer Halbebene.

    Fuer k(u) = e^{u/2}/(2*sinh(u)) auf u > 0:
    k-hat(s) = int_0^inf e^{i*s*u} * e^{u/2}/(2*sinh(u)) du

    = int_0^inf e^{(1/2 + is)*u} / (e^u - e^{-u}) du / 2
    = (1/2) int_0^inf e^{(1/2 + is)*u} * sum_{n=0}^inf e^{-(2n+1)*u} du
    = (1/2) sum_{n=0}^inf 1/(2n + 1/2 - is)
    = (1/4) [psi(3/4 - is/2) - psi(1/4 - is/2)]  (Digamma-Formel)

    Wobei psi = Gamma'/Gamma die Digamma-Funktion ist.

    Die Nullstellen von 1/k-hat liegen dort wo k-hat -> infinity,
    also bei den Polen der Digamma-Funktion: 1/4 - is/2 = -n oder 3/4 - is/2 = -n.
    Das gibt s = i*(1/2 + 2n) oder s = i*(3/2 + 2n).
    Alle Polen liegen bei POSITIVEM Im(s) -> 1/k-hat hat keine Nullstellen
    in der oberen Halbebene!

    DAS WAERE DER TP-BEWEIS FUER DEN ARCHIMEDISCHEN KERN!
    """
    print("\n" + "=" * 80)
    print("FOURIER-ANALYSE: Schoenberg-Kriterium fuer TP")
    print("=" * 80)

    from scipy.special import digamma

    # Berechne k-hat numerisch
    s_values = np.linspace(-10, 10, 200)
    k_hat = np.zeros(len(s_values), dtype=complex)

    for idx, s in enumerate(s_values):
        # Numerische Integration
        u_grid = np.linspace(0.001, 30, 5000)
        du = u_grid[1] - u_grid[0]
        integrand = np.exp(1j * s * u_grid) * np.exp(u_grid / 2) / (2 * np.sinh(u_grid))
        k_hat[idx] = np.sum(integrand) * du

    print("\n  k-hat(s) fuer reelle s (sollte keine Nullstellen haben):")
    print(f"  {'s':>6} | {'Re(k-hat)':>12} | {'Im(k-hat)':>12} | {'|k-hat|':>12}")
    print(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for i in range(0, len(s_values), 20):
        s = s_values[i]
        kh = k_hat[i]
        print(f"  {s:6.2f} | {kh.real:+12.6f} | {kh.imag:+12.6f} | {abs(kh):12.6f}")

    # Teste ob |k-hat| > 0 ueberall (keine Nullstellen auf reeller Achse)
    min_abs = np.min(np.abs(k_hat))
    print(f"\n  min|k-hat(s)| auf reeller Achse = {min_abs:.6e}")
    print(f"  k-hat hat {'KEINE' if min_abs > 1e-6 else 'MOEGLICHERWEISE'} "
          f"Nullstellen auf reeller Achse")

    # Analytische Formel
    print("\n  ANALYTISCHE FORMEL:")
    print("  k-hat(s) = (1/4) * [psi(3/4 - is/2) - psi(1/4 - is/2)]")
    print("  Pole von psi bei z = 0, -1, -2, ...")
    print("  => Pole von k-hat bei: 1/4 - is/2 = -n => s = i*(1/2 + 2n)")
    print("     und: 3/4 - is/2 = -n => s = i*(3/2 + 2n)")
    print("  ALLE Pole liegen bei Im(s) > 0!")
    print("  => 1/k-hat hat keine Pole in oberer Halbebene")
    print("  => k-hat hat keine NULLSTELLEN in oberer Halbebene")
    print("  => Nach Schoenberg: k ist Polya-Frequenz-Funktion (PF_infinity)")
    print("  => k ist TOTAL POSITIV auf R!")
    print("\n  ABER: Dies gilt fuer k auf ganz R, nicht fuer die TRUNKIERUNG auf [0, 2L].")
    print("  Trunkierungen total positiver Kerne sind nicht automatisch TP.")
    print("  Und der Regularierer -2*e^{-s/2}*delta(s)*I muss separat behandelt werden.")


if __name__ == "__main__":
    t0 = time.time()
    test_arch_kernel_TP()
    test_shifted_kernel_TP()
    fourier_analysis()
    test_full_QW_positivity()
    print(f"\nTotal: {time.time()-t0:.1f}s")
