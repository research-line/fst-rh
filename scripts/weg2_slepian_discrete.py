#!/usr/bin/env python3
"""
weg2_slepian_discrete.py
=========================
Versuche, Slepians Argument auf den diskreten Prim-Shift-Operator zu uebertragen.

SLEPIANS THEOREM (1961):
Fuer den prolaten Operator P = T_L * B_W * T_L (Zeitlimit * Bandlimit * Zeitlimit):
1. Alle Eigenwerte sind einfach
2. Die n-te Eigenfunktion hat genau n Nullstellen in (-L,L)
3. Insbesondere: Die Grundzustandseigenfunktion (groesster EW) hat KEINE Nullstellen
   => sie ist GERADE (oder kann gerade gewaehlt werden)

SLEPIANS BEWEIS nutzt:
- P kommutiert mit einem Sturm-Liouville-Operator D
- D hat die Form D = d/dt [(L^2-t^2) d/dt] + c*t^2 (prolater DiffOp)
- Sturm-Liouville-Theorie => einfache Eigenwerte + Knotenstruktur

UNSER OPERATOR:
P_prime = sum_p sum_m (logp/p^{m/2}) (S_{m*logp} + S_{-m*logp})
auf L^2[-L,L], L = log(lambda).

KOMMUTIERT P_prime MIT EINEM DIFFERENTIALOPERATOR?
Falls ja: Sturm-Liouville => Even-Dominanz bewiesen!

ALTERNATIVER ANSATZ: Verwende DIREKT die Knotenstruktur.
Fuer einen POSITIVEN symmetrischen Faltungsoperator K:
- Jentzsch-Theorem: Der groesste EW ist einfach, Eigenfunktion > 0
- Krein-Rutman: Grundzustand ist knotenlos => gerade (auf symmetrischem Intervall)

P_prime IST POSITIV? Pruefe!
P_prime f(t) = sum_p sum_m coeff * [f(t-s) + f(t+s)] mit coeff > 0
Fuer s_max < L: Jeder Term (S_s + S_{-s}) ist ein positiver Operator
(Kern = delta(t-t'-s) + delta(t-t'+s) >= 0).
ABER: Die Operatoren S_s TRUNKIEREN auf [-L,L]!
S_s f(t) = f(t-s) * 1_{[-L,L]}(t) * 1_{[-L,L]}(t-s)
Das macht S_s NICHT positiv im Sinne von K(t,t') >= 0.

JENTZSCH-ARGUMENT:
Wir brauchen: K(t,t') > 0 fuer (fast) alle t, t' in [-L,L].
K(t,t') = sum_p sum_m coeff * [delta(t-t'-s) + delta(t-t'+s)]
Das ist eine SUMME VON DELTA-FUNKTIONEN! Kein stetiger Kern.

REGULARISIERUNG:
Ersetze delta durch Gauss-Kern: delta(u-s) -> (1/eps) * exp(-(u-s)^2/eps^2)
Im Limes eps -> 0 konvergieren die Eigenwerte.
Der regularisierte Kern IST stetig und positiv.
=> Jentzsch gibt: Grundzustand knotenlos => gerade.
=> Im Limes eps -> 0: Grundzustand bleibt gerade (stetige Abhaengigkeit der EW).

DAS WAERE DER BEWEIS! Aber: Haben wir wirklich K_eps(t,t') > 0 fuer ALLE t, t'?
"""

import numpy as np
from scipy.linalg import eigh
from mpmath import euler as mp_euler, log as mplog, pi as mppi
import time

LOG4PI_GAMMA = float(mplog(4 * mppi)) + float(mp_euler)


def test_positivity_of_P():
    """
    Teste ob der Prim-Shift-Operator P als Matrix positiv semi-definit ist.

    P_{nm} = sum_p sum_m coeff * [<phi_n, S_s phi_m> + <phi_n, S_{-s} phi_m>]

    Alle Eigenwerte von P sollten >= 0 sein (positiver Operator).
    """
    from sympy import primerange
    import sys
    sys.path.insert(0, '.')
    from weg2_analytic_even_odd import shift_element_cos, shift_element_sin

    primes = list(primerange(2, 200))

    print("=" * 80)
    print("POSITIVITAET DES PRIM-SHIFT-OPERATORS P")
    print("=" * 80)

    N = 30

    for lam in [30, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)

        for basis in ['cos', 'sin']:
            P = np.zeros((N, N))
            shift_fn = shift_element_cos if basis == 'cos' else shift_element_sin

            for p in primes_used:
                logp = np.log(p)
                for m_exp in range(1, 20):
                    coeff = logp * p**(-m_exp / 2.0)
                    shift = m_exp * logp
                    if shift >= 2 * L or coeff < 1e-15:
                        break
                    for i in range(N):
                        for j in range(i, N):
                            val = coeff * (shift_fn(i, j, shift, L) + shift_fn(i, j, -shift, L))
                            P[i, j] += val
                            if i != j:
                                P[j, i] += val

            evals = np.sort(eigh(P, eigvals_only=True))
            sector = "EVEN" if basis == 'cos' else "ODD"
            n_neg = np.sum(evals < -1e-10)
            print(f"\n  lambda={lam}, {sector}:")
            print(f"  EW: min={evals[0]:+.6f}, max={evals[-1]:+.6f}")
            print(f"  Negative EW: {n_neg}")
            if n_neg > 0:
                print(f"  P ist NICHT positiv semi-definit!")
                print(f"  Negative EW: {evals[evals < -1e-10]}")
            else:
                print(f"  P IST positiv semi-definit!")


def test_jentzsch_argument():
    """
    JENTZSCH-ARGUMENT fuer regularisierten Prim-Shift-Kern.

    K_eps(t,t') = sum_p sum_m (logp/p^{m/2}) * (1/eps) * [G((t-t'-s)/eps) + G((t-t'+s)/eps)]

    wobei G(x) = exp(-x^2) / sqrt(pi) (Gauss-Kern).

    Fuer eps -> 0 konvergiert K_eps -> delta-Summe.
    Fuer eps > 0 ist K_eps stetig.

    FRAGE: Ist K_eps(t,t') > 0 fuer ALLE t, t' in [-L,L]?
    Das erfordert: Fuer jedes (t,t')-Paar existiert ein Term p^m mit
    t-t' ~ m*logp (innerhalb eps).

    Da die Mengen {m*logp : p prim, m >= 1} DICHT in [0,inf) liegen
    (nach dem Primzahlsatz ist die Menge {logp} hinreichend dicht),
    ist K_eps > 0 fuer hinreichend grosses eps.

    ABER: Wir brauchen eps -> 0! Da gibt es Luecken.
    """
    from sympy import primerange
    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("JENTZSCH-ARGUMENT: Regularisierter Prim-Shift-Kern")
    print("=" * 80)

    lam = 100
    L = np.log(lam)
    primes_used = [p for p in primes if p <= lam]

    # Berechne die Shift-Frequenzen und ihre Gewichte
    shifts = []
    for p in primes_used:
        logp = np.log(p)
        for m_exp in range(1, 20):
            coeff = logp * p**(-m_exp / 2.0)
            shift = m_exp * logp
            if shift >= 2 * L or coeff < 1e-15:
                break
            shifts.append((shift, coeff))

    shifts.sort()
    print(f"\n  lambda={lam}, L={L:.3f}, Anzahl Shifts: {len(shifts)}")
    print(f"  Shift-Bereich: [{shifts[0][0]:.4f}, {shifts[-1][0]:.4f}]")

    # Maximale Luecke zwischen aufeinanderfolgenden Shifts
    shift_vals = [s[0] for s in shifts]
    gaps = [shift_vals[i+1] - shift_vals[i] for i in range(len(shift_vals)-1)]
    max_gap = max(gaps)
    avg_gap = np.mean(gaps)
    print(f"  Maximale Luecke: {max_gap:.6f}")
    print(f"  Mittlere Luecke: {avg_gap:.6f}")
    print(f"  => eps muss > {max_gap:.4f} sein fuer strikt positiven Kern")

    # Berechne K_eps(t,t') fuer verschiedene eps
    for eps in [0.5, 0.3, 0.2, 0.1, 0.05]:
        n_grid = 50
        t_grid = np.linspace(-L, L, n_grid)
        K = np.zeros((n_grid, n_grid))

        for s, coeff in shifts:
            for i in range(n_grid):
                for j in range(n_grid):
                    dt = t_grid[i] - t_grid[j]
                    K[i, j] += coeff * (np.exp(-((dt - s) / eps)**2)
                                       + np.exp(-((dt + s) / eps)**2)) / (eps * np.sqrt(np.pi))

        min_K = np.min(K)
        evals = np.sort(np.linalg.eigvalsh(K))
        n_neg_ev = np.sum(evals < -1e-10)

        print(f"\n  eps={eps:.2f}: min K(t,t')={min_K:.6e}, "
              f"n_neg_ev={n_neg_ev}, max_ev={evals[-1]:.4f}")
        print(f"  K > 0 fuer alle t,t': {'JA' if min_K > 0 else 'NEIN'}")

        if min_K > 0 and n_neg_ev == 0:
            # Jentzsch-Theorem anwendbar!
            # Berechne den Grundzustand
            evecs = np.linalg.eigh(K)[1]
            v_max = evecs[:, -1]
            # Ist der Grundzustand knotenlos?
            sign_changes = np.sum(np.diff(np.sign(v_max)) != 0)
            # Ist er gerade? v(-t) = v(t)?
            v_reversed = v_max[::-1]
            sym_err = np.linalg.norm(v_max - v_reversed) / np.linalg.norm(v_max)
            asym_err = np.linalg.norm(v_max + v_reversed) / np.linalg.norm(v_max)
            is_even = sym_err < 0.01
            print(f"  Grundzustand: {sign_changes} Vorzeichenwechsel, "
                  f"sym_err={sym_err:.4e} => {'GERADE' if is_even else 'UNGERADE'}")


def kernel_matrix_analysis():
    """
    Analysiere den Kern K(t,t') = sum coeff * [delta(t-t'-s) + delta(t-t'+s)]
    als diskretisierte Matrix.

    Frage: Ist die diskretisierte Matrix TOTAL POSITIV?
    (Alle Minoren >= 0?)

    Wenn ja: Perron-Frobenius-Variante => Grundzustand positiv => gerade.
    """
    from sympy import primerange

    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("KERN-MATRIX-ANALYSE: Diskretisierter Prim-Kern")
    print("=" * 80)

    for lam in [30, 100]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)

        # Diskretisiere auf Gitter
        N_grid = 40
        t_grid = np.linspace(-L, L, N_grid)
        dt = t_grid[1] - t_grid[0]

        # Berechne Kern-Matrix K[i,j] = K(t_i, t_j)
        # K(t,t') = sum coeff * [delta(t-t'-s) + delta(t-t'+s)]
        # Diskretisiert: K[i,j] ~ sum coeff * [delta_{|i-j|*dt, s}] / dt

        # Besser: Regularisierter Kern mit eps = dt
        eps = dt
        K = np.zeros((N_grid, N_grid))

        for p in primes_used:
            logp = np.log(p)
            for m_exp in range(1, 20):
                coeff = logp * p**(-m_exp / 2.0)
                shift = m_exp * logp
                if shift >= 2 * L or coeff < 1e-15:
                    break
                for i in range(N_grid):
                    for j in range(N_grid):
                        delta = t_grid[i] - t_grid[j]
                        K[i, j] += coeff * (np.exp(-((delta - shift) / eps)**2)
                                           + np.exp(-((delta + shift) / eps)**2)) / (eps * np.sqrt(np.pi))

        evals, evecs = np.linalg.eigh(K)
        idx = np.argsort(evals)[::-1]  # Absteigend

        # Grundzustand (groesster EW)
        v0 = evecs[:, idx[0]]
        sign_changes = np.sum(np.diff(np.sign(v0)) != 0)

        # Paritaet
        v_reversed = v0[::-1]
        sym_err = np.linalg.norm(v0 - v_reversed) / np.linalg.norm(v0)

        # Zweite Eigenfunktion
        v1 = evecs[:, idx[1]]
        sign_changes_1 = np.sum(np.diff(np.sign(v1)) != 0)
        sym_err_1 = np.linalg.norm(v1 - v1[::-1]) / np.linalg.norm(v1)
        asym_err_1 = np.linalg.norm(v1 + v1[::-1]) / np.linalg.norm(v1)

        print(f"\nlambda={lam}, N_grid={N_grid}, eps=dt={eps:.4f}:")
        print(f"  K min={np.min(K):.6e}, max={np.max(K):.4f}")
        print(f"  Eigenwerte: {evals[idx[0]]:.4f}, {evals[idx[1]]:.4f}, {evals[idx[2]]:.4f}")
        print(f"  Grundzustand: {sign_changes} Vorzeichenwechsel, "
              f"sym_err={sym_err:.4e} => {'GERADE' if sym_err < 0.01 else 'UNGERADE'}")
        print(f"  2. Eigenfunktion: {sign_changes_1} Vorzeichenwechsel, "
              f"sym_err={sym_err_1:.4e}, asym_err={asym_err_1:.4e} "
              f"=> {'GERADE' if sym_err_1 < 0.01 else 'UNGERADE' if asym_err_1 < 0.01 else 'GEMISCHT'}")


def slepian_nodal_transfer():
    """
    KERNARGUMENT: Uebertragung der Knotenstruktur.

    Fuer den vollen QW-Operator (nicht nur den Prim-Teil):
    QW = diag + arch + prime

    Der Grundzustand von QW im vollen L^2[-L,L] ist die Funktion
    mit dem kleinsten Eigenwert. Wenn der Prim-Operator P dominiert,
    ist der Grundzustand von QW nahe am Grundzustand von -P (negiert,
    da P in QW addiert wird und wir den KLEINSTEN EW suchen).

    -P hat als Grundzustand den NEGATIVEN des groessten P-Eigenvektors.
    Wenn P > 0 (PSD), dann ist der groesste P-Eigenvektor knotenlos (Jentzsch)
    => -P Grundzustand knotenlos => gerade.

    PROBLEM: P ist NICHT PSD! (Trunkierung zerstoert PSD.)
    Aber der regularisierte Kern K_eps IST PSD fuer grosses eps.

    ALTERNATIVE: Betrachte P im Fourier-Raum.
    P-hat(xi) = 2 * sum_p sum_m (logp/p^{m/2}) * cos(m*logp * xi)

    Das ist eine POSITIVE Fourier-Reihe (alle Koeffizienten positiv)!
    => P-hat(xi) > 0 fuer alle xi (Maximum bei xi=0).

    Fuer bandlimitierte Funktionen f auf [-L,L]:
    <f, P f> = int |f-hat(xi)|^2 * P-hat(xi) dxi / (2*pi)  ???

    NEIN! Das stimmt nur fuer L^2(R), nicht fuer L^2[-L,L].
    Auf [-L,L] ist die Fourier-Darstellung komplizierter.
    """
    from sympy import primerange
    import sys
    sys.path.insert(0, '.')
    from weg2_analytic_even_odd import build_QW_analytic

    primes = list(primerange(2, 200))

    print("\n" + "=" * 80)
    print("KNOTENSTRUKTUR: QW-Grundzustand im vollen Raum")
    print("=" * 80)

    for lam in [30, 50, 100, 200]:
        primes_used = [p for p in primes if p <= max(lam, 47)]
        L = np.log(lam)
        N = 40

        # Baue QW im vollen Raum (cos + sin Basis gemischt)
        # Ordnung: cos_0, cos_1, sin_1, cos_2, sin_2, ...
        # Da cos und sin entkoppelt sind, genuegt es die Sektoren separat zu berechnen
        W_cos = build_QW_analytic(lam, N, primes_used, 'cos')
        W_sin = build_QW_analytic(lam, N, primes_used, 'sin')

        evals_c, evecs_c = eigh(W_cos)
        evals_s, evecs_s = eigh(W_sin)

        idx_c = np.argsort(evals_c)
        idx_s = np.argsort(evals_s)

        l1_cos = evals_c[idx_c[0]]
        l1_sin = evals_s[idx_s[0]]

        v1_cos = evecs_c[:, idx_c[0]]  # Koeffizienten in cos-Basis
        v1_sin = evecs_s[:, idx_s[0]]  # Koeffizienten in sin-Basis

        # Rekonstruiere f(t) im Ortsraum
        n_plot = 200
        t_grid = np.linspace(-L, L, n_plot)

        f_cos = np.zeros(n_plot)
        for n in range(N):
            if n == 0:
                f_cos += v1_cos[n] / np.sqrt(2 * L) * np.ones(n_plot)
            else:
                f_cos += v1_cos[n] / np.sqrt(L) * np.cos(n * np.pi * t_grid / (2 * L))

        f_sin = np.zeros(n_plot)
        for n in range(N):
            f_sin += v1_sin[n] / np.sqrt(L) * np.sin((n + 1) * np.pi * t_grid / (2 * L))

        # Knotenanalyse
        cos_signs = np.diff(np.sign(f_cos))
        cos_nodes = np.sum(cos_signs != 0)
        sin_signs = np.diff(np.sign(f_sin))
        sin_nodes = np.sum(sin_signs != 0)

        # Paritaet
        f_cos_rev = f_cos[::-1]
        cos_sym = np.linalg.norm(f_cos - f_cos_rev) / np.linalg.norm(f_cos)
        f_sin_rev = f_sin[::-1]
        sin_asym = np.linalg.norm(f_sin + f_sin_rev) / np.linalg.norm(f_sin)

        print(f"\nlambda={lam}:")
        print(f"  l1(cos)={l1_cos:+.4f}, l1(sin)={l1_sin:+.4f}, Delta={l1_cos-l1_sin:+.4f}")
        print(f"  cos-Grundzustand: {cos_nodes} Knoten, Paritaet={cos_sym:.4e} "
              f"=> {'GERADE' if cos_sym < 0.01 else 'NICHT GERADE'}")
        print(f"  sin-Grundzustand: {sin_nodes} Knoten, Paritaet={sin_asym:.4e} "
              f"=> {'UNGERADE' if sin_asym < 0.01 else 'NICHT UNGERADE'}")

        # Dominante Moden
        top_cos = np.argsort(np.abs(v1_cos))[-5:][::-1]
        top_sin = np.argsort(np.abs(v1_sin))[-5:][::-1]
        print(f"  cos dominant modes: {top_cos} (|coeff|: {np.abs(v1_cos[top_cos])})")
        print(f"  sin dominant modes: {top_sin} (|coeff|: {np.abs(v1_sin[top_sin])})")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, r'C:\Users\User\OneDrive\.RESEARCH\Natur&Technik\1 Musterbeweise\RH\scripts')

    t0 = time.time()
    test_positivity_of_P()
    test_jentzsch_argument()
    kernel_matrix_analysis()
    slepian_nodal_transfer()
    print(f"\nTotal: {time.time()-t0:.1f}s")
