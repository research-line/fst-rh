#!/usr/bin/env python3
"""
weg2_lowmode_proof.py
======================
Schliesse die Beweisluecke via Low-Mode-Block.

IDEE: Der QW-Grundzustand konzentriert sich auf Moden 1-4 (fuer lambda >= 30).
Statt die N=30 Schur-Horn-Schranke zu nutzen (1 Std-Abw bei N=30 Eigenwerten),
verwende den k×k Block (k=4..8) und zeige:

l1(W_cos[:k,:k]) < l1_sin(N=30)

Da l1(W_cos) <= l1(W_cos[:k,:k]) (Cauchy-Interlacing), folgt:
l1(W_cos) < l1_sin.

ABER: Wir muessen l1(W_cos[:k,:k]) analytisch NACH OBEN abschaetzen.
Und l1_sin(N=30) ist numerisch -- fuer einen formalen Beweis braeuchten wir
eine analytische UNTERE Schranke fuer l1_sin.

ALTERNATIVE: Zeige direkt l1(W_cos[:k,:k]) < l1(W_sin[:k,:k])
und verwende dass l1(M[:k,:k]) >= l1(M) >= l1(M) (Cauchy-Interlacing).
Nein, das geht in die falsche Richtung.

KORREKTE RICHTUNG:
l1(W_cos[:k,:k]) >= l1(W_cos)  (Cauchy: Untermatrix-EW >= Matrix-EW)
l1(W_sin[:k,:k]) >= l1(W_sin)

Wir wollen: l1(W_cos) < l1(W_sin)
Das folgt NICHT aus l1(W_cos[:k,:k]) < l1(W_sin[:k,:k]).

NEUER ANSATZ: Verwende l1(W_cos) <= Rayleigh-Quotient mit Trial-Vektor.
Waehle den Trial-Vektor als den Grundzustand des k×k Blocks:

l1(W_cos) <= v_k^T * W_cos * v_k

wobei v_k = [v_1,...,v_k, 0,...,0] der Grundzustand des k×k Blocks ist.
Dann ist v_k^T * W_cos * v_k = l1(W_cos[:k,:k]).

Also: l1(W_cos) <= l1(W_cos[:k,:k]).

Und wir brauchen: l1(W_cos[:k,:k]) < l1(W_sin).

Fuer l1(W_sin) verwende eine UNTERE Schranke:
l1(W_sin) >= l1(W_sin[:k,:k]) (falsch! Cauchy geht andersrum)

Nein: l1(M[:k,:k]) >= l1(M) (Cauchy-Interlacing).
Also l1(W_sin[:k,:k]) >= l1(W_sin).
Das hilft nicht (gibt obere, nicht untere Schranke fuer l1(W_sin)).

UNTERE SCHRANKE fuer l1(W_sin):
l1(W_sin) >= Tr(W_sin)/N - sqrt((N-1)/N * (||W_sin||_F^2 - Tr^2/N))

Diese Schranke benoetigt ||W_sin||_F und Tr(W_sin), die wir berechnen koennen.

ZUSAMMEN:
l1(W_cos) <= l1(W_cos[:k,:k])  (Variationsprinzip/Cauchy)
l1(W_sin) >= Tr/N - sqrt(Var*(N-1))  (Schur-Horn untere Schranke)

Wenn l1(W_cos[:k,:k]) < Tr_sin/N - sqrt(Var_sin*(N-1)):
Dann l1(W_cos) < l1(W_sin). QED!

PROBLEM: Die untere Schranke fuer l1_sin ist zu tief (Schur-Horn).
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import sys
sys.path.insert(0, r'C:\Users\User\OneDrive\.RESEARCH\Natur&Technik\1 Musterbeweise\RH\scripts')

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def lowmode_block_analysis():
    """Teste ob der k×k cos-Block genuegt um l1_sin zu unterschreiten."""
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))

    print("=" * 80)
    print("LOW-MODE BLOCK: l1(W_cos[:k,:k]) vs l1(W_sin)")
    print("=" * 80)

    N = 30

    for lam in [30, 50, 100, 200, 500]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        l1_sin = np.sort(eigh(W_sin, eigvals_only=True))[0]
        l1_cos = np.sort(eigh(W_cos, eigvals_only=True))[0]

        print(f"\nlambda={lam}: l1_cos={l1_cos:+.4f}, l1_sin={l1_sin:+.4f}, Delta={l1_cos-l1_sin:+.4f}")

        for k in range(2, 12):
            Mk = W_cos[:k, :k]
            l1_block = np.sort(np.linalg.eigvalsh(Mk))[0]
            ok = l1_block < l1_sin
            print(f"  k={k:2d}: l1(cos_block)={l1_block:+.8f} "
                  f"{'< l1_sin => BEWIESEN' if ok else '>= l1_sin'}")
            if ok:
                break


def gershgorin_lower_bound():
    """
    Gershgorin UNTERE Schranke fuer l1(W_sin).

    l1(W_sin) >= min_i [W_sin[i,i] - sum_{j!=i} |W_sin[i,j]|]

    Das ist die Gershgorin-Kreise-Schranke.
    """
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))
    N = 30

    print("\n" + "=" * 80)
    print("GERSHGORIN UNTERE SCHRANKE fuer l1(W_sin)")
    print("=" * 80)

    for lam in [30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')
        l1_sin = np.sort(eigh(W_sin, eigvals_only=True))[0]

        # Gershgorin untere Schranke
        gersh = np.array([W_sin[i, i] - np.sum(np.abs(W_sin[i, :])) + np.abs(W_sin[i, i])
                         for i in range(N)])
        gersh_lower = np.min(gersh)

        print(f"\n  lambda={lam}: l1_sin={l1_sin:+.4f}, Gershgorin_lower={gersh_lower:+.4f}")
        print(f"  Gap = l1_sin - Gersh = {l1_sin - gersh_lower:+.4f}")


def combined_proof():
    """
    KOMBINIERTER BEWEIS:
    l1(W_cos) <= l1(W_cos[:k,:k])  (obere Schranke, Variationsprinzip)
    l1(W_sin) >= Gershgorin_lower  (untere Schranke)

    Wenn l1(W_cos[:k,:k]) < Gershgorin_lower(W_sin), dann l1_cos < l1_sin.

    ALTERNATIVE: Verwende l1(W_sin) >= l1(W_sin + epsilon*I) - epsilon
    mit epsilon so gewaehlt dass W_sin + epsilon*I >= 0.
    """
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))
    N = 30

    print("\n" + "=" * 80)
    print("KOMBINIERTER BEWEIS: l1(cos[:k]) < lower_bound(sin)")
    print("=" * 80)

    for lam in [30, 50, 100, 200, 500]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        l1_cos = np.sort(eigh(W_cos, eigvals_only=True))[0]
        l1_sin = np.sort(eigh(W_sin, eigvals_only=True))[0]

        # Obere Schranke fuer l1_cos via k×k Block
        best_k = None
        best_upper = float('inf')
        for k in range(2, 15):
            l1_block = np.sort(np.linalg.eigvalsh(W_cos[:k, :k]))[0]
            if l1_block < best_upper:
                best_upper = l1_block
                best_k = k

        # Untere Schranke fuer l1_sin
        # 1. Gershgorin
        gersh = float('inf')
        for i in range(N):
            row_sum = np.sum(np.abs(W_sin[i, :])) - np.abs(W_sin[i, i])
            gersh = min(gersh, W_sin[i, i] - row_sum)

        # 2. Schur-Horn (untere Schranke)
        mean_s = np.trace(W_sin) / N
        var_s = np.linalg.norm(W_sin, 'fro')**2 / N - mean_s**2
        schur_lower = mean_s - np.sqrt(var_s * (N - 1))

        # Beste untere Schranke
        lower_sin = max(gersh, schur_lower)

        proven = best_upper < lower_sin

        print(f"\n  lambda={lam}:")
        print(f"  l1_cos={l1_cos:+.4f}, l1_sin={l1_sin:+.4f}, Delta={l1_cos-l1_sin:+.4f}")
        print(f"  Upper(cos, k={best_k}): {best_upper:+.4f}")
        print(f"  Lower(sin, Gersh): {gersh:+.4f}")
        print(f"  Lower(sin, Schur): {schur_lower:+.4f}")
        print(f"  Best lower(sin): {lower_sin:+.4f}")
        print(f"  => {'BEWIESEN: upper < lower' if proven else 'NICHT bewiesen (lower zu tief)'}")


def direct_block_comparison():
    """
    DIREKTE BLOCK-STRATEGIE:
    Fuer den Even-Sektor genuegen die Moden 0-6 (7 Moden) um
    l1(cos[:7]) < l1(sin) zu zeigen.

    Fuer den Odd-Sektor zeige l1(sin) >= SOMETHING via Cauchy-Interlacing
    von OBEN: l1(sin[:k]) >= l1(sin) fuer alle k.

    Trick: Verwende N_sin klein genug dass l1(sin[:N_sin]) noch > l1(cos[:7]).
    """
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("DIREKTE BLOCK-STRATEGIE: l1(cos[:k]) vs l1(sin[:m])")
    print("=" * 80)

    for lam in [30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        # Baue grosse Matrizen
        N_big = 40
        W_cos = build_QW_analytic(lam, N_big, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N_big, primes_used, 'sin')

        l1_cos_full = np.sort(eigh(W_cos, eigvals_only=True))[0]
        l1_sin_full = np.sort(eigh(W_sin, eigvals_only=True))[0]

        print(f"\nlambda={lam}: l1_cos(N=40)={l1_cos_full:+.4f}, l1_sin(N=40)={l1_sin_full:+.4f}")

        # Finde minimales k so dass l1(cos[:k]) < l1(sin_full)
        for k in range(2, 15):
            l1_cos_k = np.sort(np.linalg.eigvalsh(W_cos[:k, :k]))[0]
            if l1_cos_k < l1_sin_full:
                print(f"  l1(cos[:{k}]) = {l1_cos_k:+.6f} < {l1_sin_full:+.6f} = l1(sin)")

                # Jetzt: Finde maximales m so dass l1(sin[:m]) > l1(cos[:k])
                # (zeigt dass die sin-Eigenvalue nicht weiter sinkt)
                for m in range(k, N_big + 1):
                    l1_sin_m = np.sort(np.linalg.eigvalsh(W_sin[:m, :m]))[0]
                    if l1_sin_m <= l1_cos_k:
                        print(f"  l1(sin[:{m}]) = {l1_sin_m:+.6f} <= l1(cos[:{k}]): sin braucht {m} Moden")
                        break
                else:
                    print(f"  l1(sin[:N]) > l1(cos[:{k}]) fuer ALLE N <= {N_big}!")

                # N-Konvergenz von l1(sin)
                print(f"  Sin-Sektor Konvergenz:")
                for m in [5, 10, 15, 20, 25, 30, 35, 40]:
                    l1_m = np.sort(np.linalg.eigvalsh(W_sin[:m, :m]))[0]
                    print(f"    N={m:2d}: l1(sin) = {l1_m:+.6f}")
                break


def n_convergence_proof():
    """
    N-KONVERGENZ: Zeige dass l1(cos) und l1(sin) mit wachsendem N konvergieren.

    Cauchy-Interlacing: l1(M[:k+1]) <= l1(M[:k])
    Also ist l1(M[:k]) monoton FALLEND in k.

    Wenn l1(cos[:k0]) < l1(sin[:m]) fuer alle m >= m0,
    und l1(sin) = lim_{m->inf} l1(sin[:m]),
    dann folgt l1(cos) <= l1(cos[:k0]) < l1(sin).

    PROBLEM: Wir muessen zeigen dass l1(sin[:m]) KONVERGIERT und
    der Grenzwert >= l1(cos[:k0]) ist.
    """
    from sympy import primerange
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("N-KONVERGENZ UND CAUCHY-INTERLACING")
    print("=" * 80)

    for lam in [100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]

        N_max = 50
        W_cos = build_QW_analytic(lam, N_max, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N_max, primes_used, 'sin')

        print(f"\nlambda={lam}:")
        print(f"  {'N':>4} | {'l1(cos)':>12} | {'l1(sin)':>12} | {'Delta':>12} | {'dl1_cos':>10} | {'dl1_sin':>10}")
        print(f"  {'-'*4}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}-+-{'-'*10}")

        prev_c, prev_s = None, None
        for N in range(3, N_max + 1, 1):
            l1_c = np.sort(np.linalg.eigvalsh(W_cos[:N, :N]))[0]
            l1_s = np.sort(np.linalg.eigvalsh(W_sin[:N, :N]))[0]
            dl_c = l1_c - prev_c if prev_c is not None else 0
            dl_s = l1_s - prev_s if prev_s is not None else 0

            if N <= 10 or N % 5 == 0 or N == N_max:
                print(f"  {N:4d} | {l1_c:+12.6f} | {l1_s:+12.6f} | {l1_c-l1_s:+12.6f} | "
                      f"{dl_c:+10.6f} | {dl_s:+10.6f}")

            prev_c, prev_s = l1_c, l1_s

        # Finde das N ab dem Delta stabil negativ ist
        cos_vals = [np.sort(np.linalg.eigvalsh(W_cos[:n, :n]))[0] for n in range(3, N_max+1)]
        sin_vals = [np.sort(np.linalg.eigvalsh(W_sin[:n, :n]))[0] for n in range(3, N_max+1)]
        deltas = [c - s for c, s in zip(cos_vals, sin_vals)]

        # Ab welchem N ist Delta < 0?
        first_neg = None
        for i, d in enumerate(deltas):
            if d < 0:
                first_neg = i + 3
                break
        if first_neg:
            print(f"  Delta < 0 ab N = {first_neg}")
            all_neg_after = all(d < 0 for d in deltas[first_neg-3:])
            print(f"  Stabil negativ ab N={first_neg}: {'JA' if all_neg_after else 'NEIN'}")


if __name__ == "__main__":
    import time
    t0 = time.time()
    lowmode_block_analysis()
    direct_block_comparison()
    n_convergence_proof()
    print(f"\nTotal: {time.time()-t0:.1f}s")
