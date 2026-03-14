#!/usr/bin/env python3
"""
partA_proof_sketch.py
=====================
Verification of the analytical argument for Part A of M1'':

CLAIM: (E_sin - E_cos) / D(lambda) → 0 as lambda → ∞

PROOF SKETCH:
1. Each E can be written as E = sum_j |B_j|^2 / gap_j
2. B_j = sum_p (logp/√p) * S(lead, j, logp, L)  [+ higher prime powers]
3. gap_j = W_jj - W_lead
4. Both B_j^2 and gap_j scale as √lambda (= e^{L/2})
5. So each term B_j^2/gap_j scales as √lambda
6. Therefore E = C(L) * √lambda where C(L) = sum_j C_j(L)
7. C_j(L) involves S(n,j,logp,L)^2 / (W_jj - W_lead)/√lambda

KEY STEP: Show c_sin(L) - c_cos(L) → 0 as L → ∞.

The overlap S(n,j,δ,L) for δ << L converges to a universal limit:
  S_cos(n,j,δ,L) → cos(n*π*δ/(2L)) * cos(j*π*δ/(2L))  [approximately]
  S_sin(n,j,δ,L) → sin((n+1)*π*δ/(2L)) * sin((j+1)*π*δ/(2L))  [approximately]

The DIFFERENCE S_cos(1,j,...) - S_sin(0,j,...) vanishes for small δ/L because
both bases become locally equivalent.

This script verifies the overlap convergence numerically.
"""
import numpy as np
from sympy import primerange
import sys


def S_cos(n, m, delta, L):
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
    print("PART A PROOF SKETCH: Overlap convergence as L → ∞")
    print("=" * 80)

    # For fixed prime p and mode j, study how S_cos(1,j,logp,L) and S_sin(0,j,logp,L)
    # converge as L grows

    primes_test = [2, 3, 5, 7, 11]
    Ls = [4, 5, 6, 7, 8, 9, 10, 12, 15, 20]
    modes = [0, 1, 2, 3, 4]

    print("\n--- OVERLAP CONVERGENCE for leading modes ---")
    print("S_cos(1, j, logp, L) vs S_sin(0, j, logp, L)")

    for p in [2, 5, 11]:
        delta = np.log(p)
        print(f"\nPrime p = {p}, delta = log({p}) = {delta:.4f}")
        print(f"{'L':>5s} | {'d/L':>6s}", end="")
        for j in modes[:4]:
            print(f" | {'S+_1j':>8s} {'S-_0j':>8s} {'diff':>8s}", end="")
        print()
        print("-" * 120)

        for L in Ls:
            r = delta / L
            print(f"{L:5.1f} | {r:6.4f}", end="")
            for j in modes[:4]:
                sc = S_cos(1, j, delta, L) + S_cos(1, j, -delta, L)
                ss = S_sin(0, j, delta, L) + S_sin(0, j, -delta, L)
                d = sc - ss
                print(f" | {sc:8.5f} {ss:8.5f} {d:+8.5f}", end="")
            print()
        sys.stdout.flush()

    # KEY: the RATIO of overlap sums determines the energy difference
    print("\n" + "=" * 80)
    print("NORMALIZED OVERLAP DIFFERENCE: (S_cos - S_sin) * L")
    print("If this converges, then the overlap difference is O(1/L)")
    print("=" * 80)

    for p in [2, 5, 11, 23, 47]:
        delta = np.log(p)
        print(f"\np = {p}, delta = {delta:.4f}")
        print(f"{'L':>5s} | {'j=0':>12s} | {'j=1':>12s} | {'j=2':>12s} | {'j=3':>12s}")
        for L in Ls:
            vals = []
            for j in range(4):
                sc = S_cos(1, j, delta, L) + S_cos(1, j, -delta, L)
                ss = S_sin(0, j, delta, L) + S_sin(0, j, -delta, L)
                vals.append((sc - ss) * L)
            print(f"{L:5.1f} | {vals[0]:12.6f} | {vals[1]:12.6f} | {vals[2]:12.6f} | {vals[3]:12.6f}")
        sys.stdout.flush()

    # Summary: compute c_sin(L) - c_cos(L) directly
    print("\n" + "=" * 80)
    print("DIRECT COEFFICIENT DIFFERENCE: c_sin(L) - c_cos(L)")
    print("where E = c(L) * sqrt(lambda), lambda = e^L")
    print("=" * 80)

    lambdas = [100, 200, 500, 1000, 2000, 5000, 10000]
    for lam in lambdas:
        L = np.log(lam)
        primes = [int(p) for p in primerange(2, min(lam+1, 5000))]
        sl = np.sqrt(lam)

        # Build simplified diagonals and off-diags for first 10 modes
        N = 10
        diag_cos = np.zeros(N)
        diag_sin = np.zeros(N)
        for n in range(N):
            # Diagonal = LOG4PI_GAMMA + sum_p 2*logp/sqrt(p) * S(n,n,logp,L) [m=1 only]
            diag_cos[n] = LOG4PI_GAMMA_F
            diag_sin[n] = LOG4PI_GAMMA_F
            for p in primes:
                logp = np.log(p)
                coeff = logp / np.sqrt(p)
                diag_cos[n] += 2 * coeff * S_cos(n, n, logp, L)
                diag_sin[n] += 2 * coeff * S_sin(n, n, logp, L)

        W11 = diag_cos[1]
        W00 = diag_sin[0]

        # Simplified B and gap (m=1 prime powers only)
        cos_idx = [j for j in range(N) if j != 1]
        B_cos = np.zeros(len(cos_idx))
        gaps_cos = np.zeros(len(cos_idx))
        for k, j in enumerate(cos_idx):
            gaps_cos[k] = diag_cos[j] - W11
            for p in primes:
                logp = np.log(p)
                coeff = logp / np.sqrt(p)
                B_cos[k] += 2 * coeff * (S_cos(1, j, logp, L) + S_cos(1, j, -logp, L))

        B_sin = np.zeros(N-1)
        gaps_sin = np.zeros(N-1)
        for j in range(1, N):
            gaps_sin[j-1] = diag_sin[j] - W00
            for p in primes:
                logp = np.log(p)
                coeff = logp / np.sqrt(p)
                B_sin[j-1] += 2 * coeff * (S_sin(0, j, logp, L) + S_sin(0, j, -logp, L))

        E_cos = np.sum(B_cos**2 / gaps_cos)
        E_sin = np.sum(B_sin**2 / gaps_sin)
        diff = E_sin - E_cos
        c_diff = diff / sl
        D = abs(W11 - W00)

        LOG4PI_GAMMA_F = 3.2720532309274587

        print(f"  lam={lam:6d}: c_sin-c_cos = {c_diff:8.5f}, diff/D = {diff/D:.4f}, diff/D*L = {diff/D*L:.4f}")

    sys.stdout.flush()
    print("\nIf c_sin - c_cos ~ C/L^alpha, then:")
    print("  diff = (c_sin-c_cos)*sqrt(lam) ~ C*sqrt(lam)/L^alpha")
    print("  D ~ c* * sqrt(lam) / L")
    print("  diff/D ~ C/(c* * L^{alpha-1})")
    print("  For alpha > 1: diff/D → 0. QED for Part A.")
