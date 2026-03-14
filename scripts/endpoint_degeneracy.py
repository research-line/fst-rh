#!/usr/bin/env python3
"""
endpoint_degeneracy.py
======================
The gap arises from ENDPOINT DEGENERACY in the Laplace expansion.

Key insight: f(n,n; u=1) = (-1)^n / 2 for cos, (-1)^{n+1}/2 for sin.
The gaps g_j = f(j,j;1) - f(lead;1) are either 1 or 0.
When g_j = 0 (DEGENERATE), the contribution to E is O(sqrt(lam)/L)
instead of O(sqrt(lam)/L^2) — dominating the energy.

COS: degenerate mode is j=3 (further from lead)
SIN: degenerate mode is j=2 (closer to lead)
=> SIN degenerate contribution > COS degenerate contribution
=> E_sin > E_cos => EVEN DOMINANCE
"""
import numpy as np
from sympy import primerange
import sys

LOG4PI_GAMMA = 3.2720532309274587

def S_cos(n, m, d, L):
    if abs(d) > 2*L: return 0.0
    a, b = max(-L, d-L), min(L, d+L)
    if a >= b: return 0.0
    if n==0 and m==0: norm = 1/(2*L)
    elif n==0 or m==0: norm = 1/(L*np.sqrt(2))
    else: norm = 1/L
    kn, km = n*np.pi/L, m*np.pi/L
    r = 0
    for f, ph in [(kn-km, km*d), (kn+km, -km*d)]:
        if abs(f) < 1e-12: r += np.cos(ph)*(b-a)/2
        else: r += (np.sin(f*b+ph) - np.sin(f*a+ph))/(2*f)
    return norm * r

def S_sin(n, m, d, L):
    if abs(d) > 2*L: return 0.0
    a, b = max(-L, d-L), min(L, d+L)
    if a >= b: return 0.0
    norm = 1/L
    kn, km = (n+1)*np.pi/L, (m+1)*np.pi/L
    r = 0
    for f, ph, s in [(kn-km, km*d, 1), (kn+km, -km*d, -1)]:
        if abs(f) < 1e-12: r += s*np.cos(ph)*(b-a)/2
        else: r += s*(np.sin(f*b+ph) - np.sin(f*a+ph))/(2*f)
    return norm * r


if __name__ == "__main__":
    print("ENDPOINT DEGENERACY ANALYSIS")
    print("=" * 70)
    print()

    # Endpoint values f(n,n; u=1)
    print("Diagonal endpoint values f(n,n; u=1):")
    print("  COS: f_cos(n,n;1) = cos(n*pi)/2 = (-1)^n / 2")
    print("    n=0: +1/2,  n=1: -1/2,  n=2: +1/2,  n=3: -1/2,  n=4: +1/2")
    print("  SIN: f_sin(n,n;1) = cos((n+1)*pi)/2 = (-1)^(n+1) / 2")
    print("    n=0: -1/2,  n=1: +1/2,  n=2: -1/2,  n=3: +1/2,  n=4: -1/2")
    print()

    # Gaps at leading Laplace order
    print("Leading-order gaps g_j^(0) = 2*sqrt(lam) * [f(j,j;1) - f(lead;1)]:")
    print("  COS (lead=1, f=-1/2): g_0=1, g_2=1, g_3=0(DEG!), g_4=1")
    print("  SIN (lead=0, f=-1/2): g_1=1, g_2=0(DEG!), g_3=1, g_4=0(DEG!)")
    print()
    print("  COS has degenerate modes at j=3,5,7,... (odd j != 1)")
    print("  SIN has degenerate modes at j=2,4,6,... (even j)")
    print()
    print("  CRITICAL: SIN's FIRST degenerate mode is j=2 (close to lead)")
    print("            COS's FIRST degenerate mode is j=3 (further)")
    print("  => Degenerate contribution to E_sin > E_cos")
    print()

    # Numerical verification
    for lam in [200, 1000, 5000]:
        L = np.log(lam)
        sl = np.sqrt(lam)
        primes = [int(p) for p in primerange(2, lam + 1)]

        N = 6
        W_cos = LOG4PI_GAMMA * np.eye(N)
        W_sin = LOG4PI_GAMMA * np.eye(N)
        for p in primes:
            logp = np.log(p)
            for m_exp in range(1, 20):
                coeff = logp * p ** (-m_exp / 2)
                delta = m_exp * logp
                if delta >= 2 * L:
                    break
                for i in range(N):
                    for j in range(i, N):
                        sc = coeff * (S_cos(i, j, delta, L) + S_cos(i, j, -delta, L))
                        ss = coeff * (S_sin(i, j, delta, L) + S_sin(i, j, -delta, L))
                        W_cos[i, j] += sc; W_cos[j, i] = W_cos[i, j]
                        W_sin[i, j] += ss; W_sin[j, i] = W_sin[i, j]

        W11 = W_cos[1, 1]
        W00 = W_sin[0, 0]
        D = abs(W11 - W00)

        print(f"lambda={lam:5d}, L={L:.2f}, D={D:.2f}")

        # COS: contributions by mode
        E_cos_total = 0
        for j in [0, 2, 3, 4]:
            g = W_cos[j, j] - W11
            B = W_cos[1, j]
            E_j = B ** 2 / g if abs(g) > 0.01 else 0
            E_cos_total += E_j
            deg = "DEG" if abs(g / sl) < 0.1 else "   "
            print(f"  COS j={j}: g={g:8.2f} g/sl={g/sl:.3f} |B|={abs(B):7.3f} B^2/g={E_j:7.3f} {deg}")

        # SIN: contributions by mode
        E_sin_total = 0
        for j in [1, 2, 3, 4]:
            g = W_sin[j, j] - W00
            B = W_sin[0, j]
            E_j = B ** 2 / g if abs(g) > 0.01 else 0
            E_sin_total += E_j
            deg = "DEG" if abs(g / sl) < 0.1 else "   "
            print(f"  SIN j={j}: g={g:8.2f} g/sl={g/sl:.3f} |B|={abs(B):7.3f} B^2/g={E_j:7.3f} {deg}")

        diff = E_sin_total - E_cos_total
        print(f"  E_sin={E_sin_total:.3f} E_cos={E_cos_total:.3f} diff={diff:+.3f} diff/D={diff/D:.4f}")
        print()

    print("=" * 70)
    print("MECHANISM OF EVEN DOMINANCE:")
    print("  1. Both sectors have non-degenerate modes with g ~ 2*sqrt(lam)")
    print("     contributing B^2/g ~ sqrt(lam)/L^2 to E (these are comparable)")
    print("  2. SIN has degenerate mode j=2 with g ~ O(sqrt(lam)/L)")
    print("     contributing B^2/g ~ sqrt(lam)/L (LARGER)")
    print("  3. COS has degenerate mode j=3 with g ~ O(sqrt(lam)/L)")
    print("     contributing B^2/g ~ sqrt(lam)/L (also larger, but smaller B)")
    print("  4. NET: E_sin > E_cos because SIN's degenerate mode (j=2) has")
    print("     LARGER coupling |B_sin_2| than COS's degenerate mode (j=3)")
    print("  5. This excess E_sin - E_cos is O(sqrt(lam)/L), but D is also")
    print("     O(sqrt(lam)/L), so the ratio rho = (E_sin-E_cos)/D = O(1)")
    print("  6. The fact that rho < 1 (and decreasing) is the content of M1")
