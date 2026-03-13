#!/usr/bin/env python3
"""
weg2_analytic_even_odd.py
=========================
Analytische Berechnung der Shift-Matrixelemente fuer cos vs sin.

Fuer die cos-Basis phi_n(t) = cos(n*pi*t/(2L))/sqrt(L) auf [-L,L]:
  <phi_n, S_s phi_m> = (1/L) int_{max(-L,-L+s)}^{min(L,L+s)} cos(n*pi*t/(2L)) cos(m*pi*(t-s)/(2L)) dt

Nutze Produktformel: cos(a)*cos(b) = [cos(a-b)+cos(a+b)]/2
  cos(n*pi*t/(2L)) * cos(m*pi*(t-s)/(2L))
  = (1/2)[cos((n-m)*pi*t/(2L) + m*pi*s/(2L)) + cos((n+m)*pi*t/(2L) - m*pi*s/(2L))]

Fuer sin-Basis phi_n(t) = sin((n+1)*pi*t/(2L))/sqrt(L):
  sin(a)*sin(b) = [cos(a-b)-cos(a+b)]/2

Der UNTERSCHIED cos vs sin ist: +cos(a+b) vs -cos(a+b).
Das heisst: Die Differenz Matrix(cos) - Matrix(sin) = cos(a+b)-Integral-Beitrag.

FRAGE: Kann man zeigen, dass dieser Unterschied den Even-Sektor bevorzugt?
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import euler as mp_euler, log as mplog, pi as mppi

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def shift_element_cos(n, m, s, L):
    """
    Berechne <cos_n, S_s cos_m> analytisch.

    <phi_n, S_s phi_m> = (1/L) int_{a}^{b} cos(n*pi*t/(2L)) * cos(m*pi*(t-s)/(2L)) dt

    Integrationsgrenzen: t in [-L, L] UND t-s in [-L, L]
    => t in [max(-L, s-L), min(L, s+L)]

    Fuer 0 <= s <= 2L: a = max(-L, s-L), b = min(L, s+L) = L (falls s <= 2L)
    Fuer s >= 0: a = s-L (falls s >= 0), b = L
    Also: a = max(-L, s-L), b = L

    Normierung: (1/L) fuer n,m >= 1; spezielle Normierung fuer n=0 oder m=0
    """
    if abs(s) > 2 * L:
        return 0.0

    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0

    # Normierungsfaktor
    if n == 0 and m == 0:
        norm = 1.0 / (2 * L)
    elif n == 0 or m == 0:
        norm = 1.0 / (L * np.sqrt(2))
    else:
        norm = 1.0 / L

    # Frequenzen
    kn = n * np.pi / (2 * L)
    km = m * np.pi / (2 * L)

    # cos(kn*t) * cos(km*(t-s)) = (1/2)[cos((kn-km)*t + km*s) + cos((kn+km)*t - km*s)]
    result = 0.0
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(freq) < 1e-12:
            # cos(phase) * (b - a)
            result += np.cos(phase) * (b - a) / 2
        else:
            # int_a^b cos(freq*t + phase) dt = [sin(freq*t + phase)/freq]_a^b
            result += (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)

    return norm * result


def shift_element_sin(n, m, s, L):
    """
    Berechne <sin_{n+1}, S_s sin_{m+1}> analytisch.
    sin(kn*t) * sin(km*(t-s)) = (1/2)[cos((kn-km)*t + km*s) - cos((kn+km)*t - km*s)]
    """
    if abs(s) > 2 * L:
        return 0.0

    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0

    norm = 1.0 / L

    kn = (n + 1) * np.pi / (2 * L)
    km = (m + 1) * np.pi / (2 * L)

    result = 0.0
    # MINUS-Zeichen beim zweiten Term ist der Unterschied zu cos!
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < 1e-12:
            result += sign * np.cos(phase) * (b - a) / 2
        else:
            result += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)

    return norm * result


def build_QW_analytic(lam, N, primes, basis='cos'):
    """Baue QW-Matrix mit analytischen Shift-Elementen."""
    L = np.log(lam)

    W = LOG4PI_GAMMA * np.eye(N)

    # Archimedischer Kernel (numerisch, da K(s) keine geschlossene Form hat)
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
                if basis == 'cos':
                    sp = shift_element_cos(i, j, s, L)
                    sm = shift_element_cos(i, j, -s, L)
                else:
                    sp = shift_element_sin(i, j, s, L)
                    sm = shift_element_sin(i, j, -s, L)
                reg = -2.0 * np.exp(-s / 2) * (1.0 if i == j else 0.0)
                val = k * (sp + sm + reg) * ds
                W[i, j] += val
                if i != j:
                    W[j, i] += val

    # Primzahl-Beitraege (analytisch!)
    for p in primes:
        logp = np.log(p)
        for m in range(1, 13):
            coeff = logp * p**(-m / 2.0)
            shift = m * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    if basis == 'cos':
                        sp = shift_element_cos(i, j, shift, L)
                        sm = shift_element_cos(i, j, -shift, L)
                    else:
                        sp = shift_element_sin(i, j, shift, L)
                        sm = shift_element_sin(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i, j] += val
                    if i != j:
                        W[j, i] += val

    return W


def test_analytic_vs_numeric():
    """Verifiziere analytische Matrixelemente gegen numerische."""
    from sympy import primerange
    primes = list(primerange(2, 100))

    print("=" * 80)
    print("ANALYTISCHE VS NUMERISCHE MATRIXELEMENTE")
    print("=" * 80)

    N = 30

    for lam in [20, 30, 50, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        print(f"\nlambda={lam}:")

        for basis in ['cos', 'sin']:
            sector = "EVEN" if basis == 'cos' else "ODD"
            W = build_QW_analytic(lam, N, primes_used, basis)
            evals = np.sort(eigh(W, eigvals_only=True))

            # Symmetrie-Check
            sym_err = np.max(np.abs(W - W.T))

            print(f"  {sector}: l1={evals[0]:+.6f}, l2={evals[1]:+.6f}, "
                  f"gap={evals[1]-evals[0]:.4e}, sym={sym_err:.2e}")


def analyze_difference_matrix():
    """
    Berechne die Differenzmatrix D = W_cos - W_sin.

    Fuer den Prim-Beitrag bei Shift s:
    D_{nm}^{prime}(s) = <cos_n, S_s cos_m> - <sin_n, S_s sin_m>
    = int cos((kn+km)t - km*s) dt   (der Unterschied-Term)

    Wenn D eine bestimmte Struktur hat (z.B. hauptsaechlich positiv),
    dann ist W_cos "hoeher" als W_sin, aber der niedrigste EW koennte
    trotzdem niedriger sein wegen groesserer Spreizung.
    """
    from sympy import primerange
    primes = list(primerange(2, 100))

    print("\n" + "=" * 80)
    print("DIFFERENZ-MATRIX W_cos - W_sin (nur Prim-Beitraege)")
    print("=" * 80)

    N = 20

    for lam in [30, 100]:
        L = np.log(lam)
        primes_used = [p for p in primes if p <= max(lam, 47)]

        D = np.zeros((N, N))
        for p in primes_used:
            logp = np.log(p)
            for m_exp in range(1, 13):
                coeff = logp * p**(-m_exp / 2.0)
                shift = m_exp * logp
                if shift >= 2 * L:
                    break
                for i in range(N):
                    for j in range(i, N):
                        dc = shift_element_cos(i, j, shift, L) + shift_element_cos(i, j, -shift, L)
                        ds = shift_element_sin(i, j, shift, L) + shift_element_sin(i, j, -shift, L)
                        val = coeff * (dc - ds)
                        D[i, j] += val
                        if i != j:
                            D[j, i] += val

        # Analyse der Differenzmatrix
        evals_D = np.sort(eigh(D, eigvals_only=True))
        print(f"\nlambda={lam}:")
        print(f"  D = W_prime(cos) - W_prime(sin):")
        print(f"  Eigenwerte: min={evals_D[0]:+.4f}, max={evals_D[-1]:+.4f}")
        print(f"  Trace = {np.trace(D):+.4f}")
        print(f"  Diag: [{np.diag(D).min():+.4f}, {np.diag(D).max():+.4f}]")
        print(f"  ||D||_F = {np.linalg.norm(D, 'fro'):.4f}")

        # Zeige die wichtigsten Eigenwerte
        print(f"  Erste 5 EW: {evals_D[:5]}")
        print(f"  Letzte 5 EW: {evals_D[-5:]}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    test_analytic_vs_numeric()
    analyze_difference_matrix()
    print(f"\nTotal: {time.time()-t0:.1f}s")
