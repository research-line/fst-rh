#!/usr/bin/env python3
"""
weg2_connes_Alambda_v2.py
=========================
Phase 7: Verbesserte Implementierung von Connes' Operator A_lambda.

Verbesserungen gegenueber v1:
  1. Bedingung g-hat(+-i/2) = 0 implementiert (Projektion auf Unterraum)
  2. Vektorisierte Berechnung (NumPy statt Python-Schleifen)
  3. Feinere Diskretisierung (n_quad=800, n_int=500)
  4. Groessere lambda (bis 50)
  5. Mellintransformierte statt Fouriertransformierte (Connes' Konvention)

Connes' Konvention:
  Funktionen phi(u) auf R_+* mit Traeger in [lambda^{-1}, lambda]
  Umparametrisierung: t = log(u), t in [-L, L], L = log(lambda)
  Mellintransformierte: phi-hat(s) = integral phi(u) u^{-s} du/u
                       = integral f(t) e^{-st} dt  (f(t) = phi(e^t))

  Bedingung g-hat(+-i/2) = 0 bedeutet:
    integral g(u) u^{1/2} du/u = 0  UND  integral g(u) u^{-1/2} du/u = 0
    = integral f(t) e^{t/2} dt = 0  UND  integral f(t) e^{-t/2} dt = 0

  Fuer gerade f: beide Bedingungen aequivalent zu:
    integral f(t) cosh(t/2) dt = 0
"""

import numpy as np
from mpmath import mp, im, zetazero, euler as mp_euler, log as mplog, pi as mppi
from scipy.linalg import eigh
from sympy import primerange
import time

mp.dps = 30

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)

# ===========================================================================
# Vektorisierte Basis
# ===========================================================================

def make_basis_grid(N_basis, t_grid, L):
    """Berechne alle Basisfunktionen auf dem Grid (vektorisiert)."""
    n_pts = len(t_grid)
    phi = np.zeros((N_basis, n_pts))
    phi[0, :] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N_basis):
        phi[n, :] = np.cos(n * np.pi * t_grid / (2 * L)) / np.sqrt(L)
    return phi

def make_shifted_basis(N_basis, t_grid, L, shift):
    """Basisfunktionen bei t - shift, mit Support-Abschneidung."""
    t_shifted = t_grid - shift
    mask = np.abs(t_shifted) <= L
    phi = np.zeros((N_basis, len(t_grid)))
    phi[0, mask] = 1.0 / np.sqrt(2 * L)
    for n in range(1, N_basis):
        phi[n, mask] = np.cos(n * np.pi * t_shifted[mask] / (2 * L)) / np.sqrt(L)
    return phi

# ===========================================================================
# Bedingung g-hat(+-i/2) = 0
# ===========================================================================

def constraint_projector(N_basis, t_grid, L, dt):
    """
    Projiziere auf den Unterraum {f : integral f(t) cosh(t/2) dt = 0
                                  AND integral f(t) sinh(t/2) dt = 0}.

    g-hat(i/2) = integral f(t) e^{-it/2} dt = 0
    g-hat(-i/2) = integral f(t) e^{it/2} dt = 0

    Aequivalent (fuer reelles f):
      integral f(t) cos(t/2) dt = 0  =>  NEIN, das ist Fourier, nicht Mellin!

    KORREKTUR: g-hat(s) = integral phi(u) u^{-s} du/u = integral f(t) e^{-st} dt
    g-hat(i/2) = integral f(t) e^{-it/2} dt  (rein Fourier bei xi=1/2)
    g-hat(-i/2) = integral f(t) e^{it/2} dt

    Fuer reelles f: g-hat(-i/2) = conj(g-hat(i/2))
    Also: g-hat(i/2) = 0 <==> integral f(t) cos(t/2) dt = 0 AND integral f(t) sin(t/2) dt = 0

    In Basiskoeffizienten: c^T * a = 0 und c^T * b = 0
    wobei a_n = integral phi_n(t) cos(t/2) dt, b_n = integral phi_n(t) sin(t/2) dt
    """
    phi = make_basis_grid(N_basis, t_grid, L)

    # Constraint-Vektoren
    cos_half = np.cos(t_grid / 2)
    sin_half = np.sin(t_grid / 2)

    a = phi @ cos_half * dt  # (N_basis,)
    b = phi @ sin_half * dt  # (N_basis,)

    # Fuer gerade Basis (cos): sin-Integral ist ~0 (ungerade * gerade = ungerade)
    # Also ist b ≈ 0 und nur a ist relevant.
    # Aber sicherheitshalber beide beruecksichtigen.

    # Gram-Schmidt auf die Constraint-Vektoren
    constraints = []
    if np.linalg.norm(a) > 1e-12:
        a_norm = a / np.linalg.norm(a)
        constraints.append(a_norm)
    if np.linalg.norm(b) > 1e-12:
        b_orth = b - np.dot(b, a_norm) * a_norm if constraints else b
        if np.linalg.norm(b_orth) > 1e-12:
            constraints.append(b_orth / np.linalg.norm(b_orth))

    if not constraints:
        return np.eye(N_basis)

    # Projektor: P = I - sum |c_i><c_i|
    P = np.eye(N_basis)
    for c in constraints:
        P -= np.outer(c, c)

    return P

# ===========================================================================
# W_p (Primstellen-Beitrag) -- vektorisiert
# ===========================================================================

def W_p_matrix(p, N_basis, phi_grid, t_grid, L, dt, M_terms=10):
    """Primstellen-Beitrag W_p (vektorisiert)."""
    logp = np.log(p)
    W = np.zeros((N_basis, N_basis))

    for m in range(1, M_terms + 1):
        coeff = logp * p**(-m / 2.0)
        shift = m * logp

        if shift >= 2 * L:
            break  # Kein Overlap mehr moeglich

        for sign in [1.0, -1.0]:
            s = sign * shift
            phi_s = make_shifted_basis(N_basis, t_grid, L, s)
            S = (phi_grid @ phi_s.T) * dt
            W += coeff * S

    return W

# ===========================================================================
# W_R (archimedischer Beitrag) -- vektorisiert
# ===========================================================================

def W_arch_matrix(N_basis, phi_grid, t_grid, L, dt, n_int=500):
    """Archimedischer Beitrag W_R (vektorisiert)."""
    W = LOG4PI_GAMMA * np.eye(N_basis)

    # Integral-Teil: integral_0^inf K(s) * (S_s + S_{-s} - 2*e^{-s/2}*I) ds
    # K(s) = 1/(2*sinh(s))
    s_max = min(2 * L, 8.0)  # Ueber 2L gibt es keinen Overlap
    s_grid = np.linspace(0.005, s_max, n_int)
    ds_int = s_grid[1] - s_grid[0]

    for s in s_grid:
        kernel = 1.0 / (2.0 * np.sinh(s))
        if kernel < 1e-15:
            continue

        phi_plus = make_shifted_basis(N_basis, t_grid, L, s)
        phi_minus = make_shifted_basis(N_basis, t_grid, L, -s)

        S_plus = (phi_grid @ phi_plus.T) * dt
        S_minus = (phi_grid @ phi_minus.T) * dt

        W += (S_plus + S_minus - 2.0 * np.exp(-s/2) * np.eye(N_basis)) * kernel * ds_int

    return W

# ===========================================================================
# Vollstaendiger Operator + Projektion
# ===========================================================================

def build_QW(lam, N_basis, primes, M_terms=10, n_quad=800, n_int=500):
    """Baue QW_lambda mit g-hat(+-i/2)=0 Projektion."""
    L = np.log(lam)
    t_grid = np.linspace(-L, L, n_quad)
    dt = t_grid[1] - t_grid[0]
    phi_grid = make_basis_grid(N_basis, t_grid, L)

    # Archimedisch
    W_arch = W_arch_matrix(N_basis, phi_grid, t_grid, L, dt, n_int)

    # Primstellen
    W_prime = np.zeros((N_basis, N_basis))
    for p in primes:
        W_prime += W_p_matrix(p, N_basis, phi_grid, t_grid, L, dt, M_terms)

    # QW = W_arch + W_prime (Connes: sum_v W_v)
    QW_full = W_arch + W_prime

    # Projektion auf Unterraum g-hat(+-i/2) = 0
    P = constraint_projector(N_basis, t_grid, L, dt)
    QW_proj = P @ QW_full @ P

    return QW_full, QW_proj, W_arch, W_prime, P

# ===========================================================================
# Mellintransformierte und Nullstellen
# ===========================================================================

def mellin_transform_zeros(evec, N_basis, lam, xi_max=150, n_xi=5000):
    """
    Berechne Mellintransformierte phi-hat(1/2 + i*xi) und finde Nullstellen.

    phi-hat(s) = integral phi(u) u^{-s} du/u = integral f(t) e^{-st} dt
    phi-hat(1/2 + i*xi) = integral f(t) e^{-(1/2+i*xi)t} dt
                        = integral f(t) e^{-t/2} e^{-i*xi*t} dt

    Also: FT von f(t)*e^{-t/2}, ausgewertet bei xi.
    """
    L = np.log(lam)
    n_fine = 8000
    t_fine = np.linspace(-L, L, n_fine)
    dt = t_fine[1] - t_fine[0]

    # Eigenfunktion rekonstruieren
    phi = make_basis_grid(N_basis, t_fine, L)
    f_vals = evec @ phi  # (n_fine,)

    # f(t) * e^{-t/2} (Mellin-Gewicht)
    f_weighted = f_vals * np.exp(-t_fine / 2)

    # FT: integral f_weighted(t) * e^{-i*xi*t} dt
    xi_grid = np.linspace(0, xi_max, n_xi)
    F_real = np.zeros(n_xi)
    F_imag = np.zeros(n_xi)

    for j, xi in enumerate(xi_grid):
        F_real[j] = np.sum(f_weighted * np.cos(xi * t_fine)) * dt
        F_imag[j] = -np.sum(f_weighted * np.sin(xi * t_fine)) * dt

    F_abs = np.sqrt(F_real**2 + F_imag**2)

    # Nullstellen = Stellen wo BEIDE Re und Im gleichzeitig ~0
    # Fuer gerade f und reelle Nullstellen: F_imag ~ 0, suche Vorzeichenwechsel in F_real
    zeros_real = []
    for j in range(len(F_real) - 1):
        if F_real[j] * F_real[j+1] < 0:
            xi0 = xi_grid[j] - F_real[j] * (xi_grid[j+1] - xi_grid[j]) / (F_real[j+1] - F_real[j])
            zeros_real.append(xi0)

    return np.array(zeros_real), xi_grid, F_real, F_imag

# ===========================================================================
# HAUPTPROGRAMM
# ===========================================================================

if __name__ == "__main__":
    print("=" * 75)
    print("PHASE 7: CONNES' OPERATOR A_lambda (verbessert)")
    print("  + g-hat(+-i/2)=0 Projektion")
    print("  + Vektorisiert, feinere Quadratur")
    print("=" * 75)

    # Zeta-Nullstellen
    print("\n  Lade 50 Zeta-Nullstellen...")
    gammas = np.array([float(im(zetazero(k))) for k in range(1, 51)])
    print(f"  gamma_1 = {gammas[0]:.6f}, gamma_50 = {gammas[-1]:.6f}")

    # Primzahlen
    all_primes = list(primerange(2, 200))

    # Parameter
    lambdas = [5, 8, 13, 20, 30, 50]
    N_BASIS = 40  # Doppelt so viel wie v1

    for lam in lambdas:
        t0 = time.time()
        L = np.log(lam)
        print(f"\n{'='*75}")
        print(f"  lambda = {lam}, L = {2*L:.4f}, Support-Laenge = {2*L:.3f}")

        # Primzahlen mit p <= lambda verwenden (plus einige groessere)
        primes_used = [p for p in all_primes if p <= max(lam, 47)]
        n_primes = len(primes_used)
        print(f"  Primzahlen: {n_primes} (bis p={primes_used[-1]})")

        # Baue Operator
        QW_full, QW_proj, W_arch, W_prime, P = build_QW(
            lam, N_BASIS, primes_used,
            M_terms=12, n_quad=800, n_int=400
        )

        # ---- Unprojiziertes Spektrum ----
        evals_full, evecs_full = eigh(QW_full)

        # ---- Projiziertes Spektrum ----
        evals_proj, evecs_proj = eigh(QW_proj)
        # Entferne die ~0-Eigenwerte die vom Projektor kommen
        # P hat Rang N_basis - n_constraints, also n_constraints EWs ~ 0
        threshold_proj = 1e-8
        active_mask = np.abs(evals_proj) > threshold_proj
        evals_active = evals_proj[active_mask]
        evecs_active = evecs_proj[:, active_mask]

        elapsed = time.time() - t0

        print(f"\n  OHNE Projektion (volles Spektrum):")
        print(f"    n_neg={np.sum(evals_full < -1e-10)}, n_pos={np.sum(evals_full > 1e-10)}")
        print(f"    lambda_min = {evals_full[0]:+.8e}")
        print(f"    lambda_2   = {evals_full[1]:+.8e}")

        print(f"\n  MIT g-hat(+-i/2)=0 Projektion:")
        print(f"    Aktive Dimension: {len(evals_active)}")
        if len(evals_active) > 0:
            n_neg_proj = np.sum(evals_active < -1e-10)
            n_pos_proj = np.sum(evals_active > 1e-10)
            print(f"    n_neg={n_neg_proj}, n_pos={n_pos_proj}")
            print(f"    lambda_min = {evals_active[0]:+.8e}")
            if len(evals_active) > 1:
                print(f"    lambda_2   = {evals_active[1]:+.8e}")
                gap = evals_active[1] - evals_active[0]
                print(f"    Luecke     = {gap:.8e}")

        # Traces
        tr_arch = np.trace(W_arch)
        tr_prime = np.trace(W_prime)
        print(f"\n  Traces: arch={tr_arch:+.4f}, prime={tr_prime:+.4f}, "
              f"total={tr_arch+tr_prime:+.4f}, ratio={tr_arch/tr_prime:.2f}" if tr_prime != 0
              else f"\n  Traces: arch={tr_arch:+.4f}, prime={tr_prime:+.4f}")

        # Mellintransformierte des kleinsten EV (projiziert)
        if len(evals_active) > 0:
            v0 = evecs_active[:, 0]  # kleinster EV im projizierten Raum
            zeros, xi_grid, F_real, F_imag = mellin_transform_zeros(v0, N_BASIS, lam)

            print(f"\n  Mellin-Nullstellen (auf krit. Linie 1/2+i*xi):")
            print(f"    Anzahl: {len(zeros)}")
            if len(zeros) > 0:
                print(f"    Erste 5: {np.array2string(zeros[:5], precision=6)}")

                # Vergleich mit Zeta-Nullstellen
                print(f"\n    {'k':>3} | {'gamma_k':>10} | {'approx':>10} | {'Delta':>10}")
                print(f"    {'-'*3}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
                for k in range(min(10, len(gammas))):
                    dists = np.abs(zeros - gammas[k])
                    if len(dists) > 0:
                        best = np.min(dists)
                        best_idx = np.argmin(dists)
                        print(f"    {k+1:3d} | {gammas[k]:10.6f} | {zeros[best_idx]:10.6f} | {best:10.6f}")

                # Imaginaerteil-Check (sollte ~0 sein bei echten Nullstellen)
                max_imag = np.max(np.abs(F_imag))
                mean_imag = np.mean(np.abs(F_imag))
                print(f"\n    Im-Teil Check: max|Im|={max_imag:.6e}, mean|Im|={mean_imag:.6e}")

        print(f"\n  Zeit: {elapsed:.1f}s")

    # FAZIT
    print(f"\n{'='*75}")
    print(f"FAZIT: PHASE 7")
    print(f"{'='*75}")
    print(f"""
  PRUEFPUNKTE (Connes' Theorem 6.1):

  1. Ist QW_lambda NEGATIV auf dem Unterraum g-hat(+-i/2)=0?
     Unter RH: JA (Weil-Positivitaet = Negativitaet von QW)

  2. Ist der NEGATIVSTE Eigenwert EINFACH?
     Theorem 6.1 braucht: isolierter, einfacher minimaler EW
     => Spektralluecke > 0

  3. Hat die Eigenfunktion eine FT mit NUR REELLEN Nullstellen?
     Theorem 6.1 garantiert das, WENN (2) erfuellt

  4. Approximieren die Nullstellen die Zeta-Nullstellen?
     Fact 6.4: k_lambda -> Xi, Rate O(lambda^{{-1/2-alpha}})
     => Hurwitz => Nullstellen konvergieren

  CONNES' OFFENES PROBLEM: Zeige (2) fuer alle lambda.
  FST-RH ANSATZ: No-Coordination + Euler-Produkt => Einfachheit?
""")
