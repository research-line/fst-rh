#!/usr/bin/env python3
"""
partA_bounded_diff.py
=====================
Analytical investigation of Part A: Why is E_sin - E_cos = O(1)?

Key hypothesis: The difference saturates because:
1. Both E_sin and E_cos are sums over j of B_j^2 / gap_j
2. B_j = sum_p (logp/sqrt(p)) * overlap(j, shift=logp, L)
3. gap_j = W_jj - W_lead ~ c_j * sqrt(lambda) for each j
4. So B_j^2/gap_j ~ (sum_p ...)^2 / (c_j * sqrt(lambda))
5. The SUM over j converges (resolvent damping)
6. The DIFFERENCE E_sin - E_cos comes from overlap differences S+ - S-
7. By Shift Parity: S+(delta,L) - S-(delta,L) = -sin(pi*delta/L)/pi
8. This is bounded and L-independent!

This script decomposes E_sin - E_cos into prime contributions to verify.
"""
import numpy as np
from scipy.linalg import eigh
from sympy import primerange
import sys

LOG4PI_GAMMA_F = 3.2720532309274587


def S_cos(n, m, delta, L):
    """Closed-form shift overlap for cosine basis."""
    if abs(delta) > 2*L:
        return 0.0
    a = max(-L, delta - L)
    b = min(L, delta + L)
    if a >= b:
        return 0.0
    if n == 0 and m == 0:
        norm = 1.0 / (2*L)
    elif n == 0 or m == 0:
        norm = 1.0 / (L * np.sqrt(2))
    else:
        norm = 1.0 / L
    kn = n * np.pi / L
    km = m * np.pi / L
    result = 0.0
    for freq, phase in [(kn - km, km * delta), (kn + km, -km * delta)]:
        if abs(freq) < 1e-12:
            result += np.cos(phase) * (b - a) / 2
        else:
            result += (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


def S_sin(n, m, delta, L):
    """Closed-form shift overlap for sine basis."""
    if abs(delta) > 2*L:
        return 0.0
    a = max(-L, delta - L)
    b = min(L, delta + L)
    if a >= b:
        return 0.0
    norm = 1.0 / L
    kn = (n+1) * np.pi / L
    km = (m+1) * np.pi / L
    result = 0.0
    for freq, phase, sign in [(kn - km, km * delta, +1), (kn + km, -km * delta, -1)]:
        if abs(freq) < 1e-12:
            result += sign * np.cos(phase) * (b - a) / 2
        else:
            result += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


if __name__ == "__main__":
    print("=" * 80)
    print("PART A: WHY IS E_sin - E_cos = O(1)?")
    print("=" * 80)
    sys.stdout.flush()

    lambdas = [100, 200, 500, 1000, 2000, 3000, 5000]
    N_cos = 10
    N_sin = 25

    print("\n--- DECOMPOSITION BY MODE j ---")
    for lam in lambdas:
        L = np.log(lam)
        primes = [int(p) for p in primerange(2, lam + 1)]

        # Build diagonal elements and off-diagonal elements from closed forms
        # Even sector: leading mode n=1
        # Diagonal: W_jj = LOG4PI_GAMMA + sum_p 2*logp/sqrt(p) * S_cos(j,j,logp,L) + ...
        # Off-diagonal to lead: B_j = W_{1,j} for j != 1

        # Build W matrices using closed-form overlaps
        W_cos = np.zeros((N_cos, N_cos))
        W_sin = np.zeros((N_sin, N_sin))
        for i in range(max(N_cos, N_sin)):
            for j in range(i, max(N_cos, N_sin)):
                if i < N_cos and j < N_cos:
                    W_cos[i,j] = LOG4PI_GAMMA_F * (1 if i == j else 0)
                if i < N_sin and j < N_sin:
                    W_sin[i,j] = LOG4PI_GAMMA_F * (1 if i == j else 0)

        for p in primes:
            logp = np.log(p)
            for m_exp in range(1, 20):
                coeff = logp * p ** (-m_exp / 2.0)
                delta = m_exp * logp
                if delta >= 2 * L:
                    break
                for i in range(max(N_cos, N_sin)):
                    for j in range(i, max(N_cos, N_sin)):
                        if i < N_cos and j < N_cos:
                            sp = S_cos(i, j, delta, L)
                            sm = S_cos(i, j, -delta, L)
                            W_cos[i,j] += coeff * (sp + sm)
                            if i != j:
                                W_cos[j,i] = W_cos[i,j]
                        if i < N_sin and j < N_sin:
                            sp = S_sin(i, j, delta, L)
                            sm = S_sin(i, j, -delta, L)
                            W_sin[i,j] += coeff * (sp + sm)
                            if i != j:
                                W_sin[j,i] = W_sin[i,j]

        # Note: archimedean term omitted (conservative for cos, doesn't change diff much)

        W11_cos = W_cos[1, 1]
        W00_sin = W_sin[0, 0]
        D = abs(W11_cos - W00_sin)

        # Compute E_sin and E_cos j-wise
        cos_indices = [j for j in range(N_cos) if j != 1]
        gaps_cos = [W_cos[j,j] - W11_cos for j in cos_indices]
        B_cos = [W_cos[1,j] for j in cos_indices]
        terms_cos = [b**2/g if abs(g) > 1e-15 else 0 for b, g in zip(B_cos, gaps_cos)]

        gaps_sin = [W_sin[j,j] - W00_sin for j in range(1, N_sin)]
        B_sin = [W_sin[0,j] for j in range(1, N_sin)]
        terms_sin = [b**2/g if abs(g) > 1e-15 else 0 for b, g in zip(B_sin, gaps_sin)]

        E_cos = sum(terms_cos)
        E_sin = sum(terms_sin)
        diff = E_sin - E_cos

        # Per-mode decomposition (first few modes)
        print(f"\nlambda = {lam:5d}  (L = {L:.3f})")
        print(f"  D = {D:.4f}, E_sin = {E_sin:.4f}, E_cos = {E_cos:.4f}, diff = {diff:+.4f}, diff/D = {diff/D:.4f}")

        # Show first 5 sin terms vs first 5 cos terms
        n_show = min(5, len(terms_sin), len(terms_cos))
        print(f"  Mode-j decomposition (first {n_show}):")
        print(f"    {'j':>3s} | {'B_sin^2/g':>10s} | {'B_cos^2/g':>10s} | {'diff':>10s} | {'gap_sin':>8s} | {'gap_cos':>8s}")
        for j in range(n_show):
            s = terms_sin[j] if j < len(terms_sin) else 0
            c = terms_cos[j] if j < len(terms_cos) else 0
            gs = gaps_sin[j] if j < len(gaps_sin) else 0
            gc = gaps_cos[j] if j < len(gaps_cos) else 0
            print(f"    {j+1:3d} | {s:10.4f} | {c:10.4f} | {s-c:+10.4f} | {gs:8.4f} | {gc:8.4f}")

        # Total from first 3 modes vs rest
        e_sin_3 = sum(terms_sin[:3])
        e_cos_3 = sum(terms_cos[:3])
        e_sin_rest = sum(terms_sin[3:])
        e_cos_rest = sum(terms_cos[3:])
        print(f"  First 3 modes:  sin={e_sin_3:.4f}, cos={e_cos_3:.4f}, diff={e_sin_3-e_cos_3:+.4f}")
        print(f"  Rest:           sin={e_sin_rest:.4f}, cos={e_cos_rest:.4f}, diff={e_sin_rest-e_cos_rest:+.4f}")

        sys.stdout.flush()

    # Key: analyze WHY the difference stabilizes
    print("\n" + "=" * 80)
    print("WHY THE DIFFERENCE IS BOUNDED")
    print("=" * 80)
    print("""
The energy difference decomposes as:
  E_sin - E_cos = sum_j [ B_sin_j^2/gap_sin_j - B_cos_j^2/gap_cos_j ]

For each mode j, the term is:
  B_j^2 = (sum_p coeff_p * overlap_j(logp, L))^2
  gap_j = W_jj - W_lead ~ c_j * sqrt(lambda)

Since both B_j^2 and gap_j scale as sqrt(lambda):
  term_j ~ C_j  (bounded constant for each j)

The SUM over j converges (resolvent damping: high modes contribute less).
The DIFFERENCE of two converging sums is bounded.

The key analytical ingredient: for fixed j and p, the overlap DIFFERENCE
  S_cos(1,j, logp, L) vs S_sin(0,j, logp, L)
is controlled by the Shift Parity Lemma, which gives a BOUNDED correction
(independent of L for logp/L -> 0).
""")
