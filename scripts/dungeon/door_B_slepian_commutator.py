"""
Tuer B: Kommutiert Slepians prolater Operator P mit der Shift-Parity-Differenzmatrix D(r)?

Slepians Operator auf L^2[-L,L] (konzentriert auf Fouriermoden |xi| < W):
  P = d/dt [ (L^2 - t^2) d/dt ] - (W^2) t^2
wo W = sqrt(lambda).

In der Standardbasis cos(n*pi*t/L)/sqrt(L) ist P tridiagonal mit bekannten Eintraegen
(Grunbaum 1981, Slepian 1961). Wir implementieren P und testen ob [P, D(r)] klein ist
im Mittel ueber r.

Wenn ja: Sturm-Liouville -> Simplicity von lambda_min -> gerade Grundzustand (A schwach!)
Wenn nein: kein direkter Slepian-Transport, aber vielleicht Stoerungsansatz moeglich.
"""

import numpy as np
from scipy.linalg import eigh
from scipy.integrate import quad

L = 1.0

def overlap_cos(n, m, s):
    """Normierte Shift-Overlap-Integrale in cos-Basis auf [-L,L]."""
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    norm_n = 1.0/np.sqrt(2*L) if n == 0 else 1.0/np.sqrt(L)
    norm_m = 1.0/np.sqrt(2*L) if m == 0 else 1.0/np.sqrt(L)
    def integrand(t):
        fn = 1.0 if n == 0 else np.cos(n*np.pi*t/L)
        fm = 1.0 if m == 0 else np.cos(m*np.pi*(t-s)/L)
        return fn * fm
    val, _ = quad(integrand, a, b, limit=200)
    return val * norm_n * norm_m

def overlap_sin(n, m, s):
    """Normierte Shift-Overlap-Integrale in sin-Basis: f_n = sin((n+1)*pi*t/L)/sqrt(L)."""
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    def integrand(t):
        return np.sin((n+1)*np.pi*t/L) * np.sin((m+1)*np.pi*(t-s)/L)
    val, _ = quad(integrand, a, b, limit=200)
    return val / L

def D_matrix(N, r):
    """Differenzmatrix D = S_cos - S_sin als NxN-Matrix (symmetrisiert fuer +r und -r)."""
    D = np.zeros((N, N))
    for n in range(N):
        for m in range(N):
            # symmetrisch: +r und -r (cos-Anteil symmetrisch, sin-Anteil symmetrisch)
            sc = (overlap_cos(n, m, r) + overlap_cos(n, m, -r)) / 2
            ss = (overlap_sin(n, m, r) + overlap_sin(n, m, -r)) / 2
            D[n, m] = sc - ss
    # Symmetrisiere numerisch
    D = (D + D.T) / 2
    return D

def slepian_P(N, W, L=1.0):
    """Slepians prolater Operator in cos-Basis, tridiagonal.

    P = d/dt[(L^2-t^2) d/dt] - W^2 t^2 auf Funktionen mit f(L)=0.
    Anwendung auf cos(n*pi*t/L)/sqrt(L): ergibt tridiagonale Matrix.

    Matrixelemente (Slepian 1961, Grunbaum):
      P_nn = -alpha_n - W^2 * (L^2/3 + L^2/(2*n^2*pi^2))  ... (vereinfacht)
      P_{n,n+2} = W^2 * coupling

    Hier: Wir verwenden die direkte Formel fuer cos-Basis auf [-L,L]:
      <cos_n | P | cos_m> = diag_part + off_diag_part
    """
    # Diagonal: (d/dt)^2 cos(n*pi*t/L) = -(n*pi/L)^2 cos(n*pi*t/L)
    # Plus boundary + t^2 terms
    P = np.zeros((N, N))
    for n in range(N):
        k_n = n * np.pi / L
        # Matrixelement <cos_n | P | cos_n>:
        # 1. Teil: -(n*pi/L)^2 * <cos_n|L^2-t^2|cos_n> normiert
        # Integral int_{-L}^{L} (L^2 - t^2) cos^2(n*pi*t/L) dt / L = L^2 - <t^2>_n
        # <t^2>_n = L^2/3 + L^2/(2*n^2*pi^2)  fuer n>=1
        if n == 0:
            t2_avg = L**2 / 3  # int t^2 dt / (2L) = L^2/3
        else:
            t2_avg = L**2 / 3 + L**2 / (2 * n**2 * np.pi**2) * (-1)**n * 2  # naeherungsweise
        # 1. Ordnung-Term
        P[n, n] = -k_n**2 * (L**2 - t2_avg) - W**2 * t2_avg

    # Nicht-Diagonal: t^2 mischt zwischen Moden
    # <cos_n | t^2 | cos_m> fuer n != m
    for n in range(N):
        for m in range(n+1, N):
            # <cos_n | t^2 | cos_m> = int t^2 cos(n*pi*t/L) cos(m*pi*t/L) dt / L (beide n,m>=1)
            norm_n = 1.0/np.sqrt(2*L) if n == 0 else 1.0/np.sqrt(L)
            norm_m = 1.0/np.sqrt(2*L) if m == 0 else 1.0/np.sqrt(L)
            def integrand(t, nn=n, mm=m):
                fn = 1.0 if nn == 0 else np.cos(nn*np.pi*t/L)
                fm = 1.0 if mm == 0 else np.cos(mm*np.pi*t/L)
                return t**2 * fn * fm
            val, _ = quad(integrand, -L, L, limit=200)
            t2_nm = val * norm_n * norm_m
            # Kopplung: -W^2 * t^2, plus eventuell Boundary-Terme
            P[n, m] = -W**2 * t2_nm
            P[m, n] = P[n, m]
    return P

def commutator_norm(P, D):
    """||[P, D]||_Frobenius / (||P||*||D||)"""
    C = P @ D - D @ P
    fP = np.linalg.norm(P)
    fD = np.linalg.norm(D)
    fC = np.linalg.norm(C)
    return fC / max(fP * fD, 1e-12)

# Test: verschiedene r-Werte und lambda-Werte
print("=== Test: kommutiert Slepians P mit D(r)? ===")
print(f"{'N':>3} {'W^2':>8} {'r':>6} {'||[P,D]||_F':>14} {'rel':>10}")
print("-" * 50)

for N in [6, 10, 15]:
    for W2 in [100.0, 1000.0]:
        W = np.sqrt(W2)
        P = slepian_P(N, W)
        for r in [0.3, 0.7, 1.0, 1.3, 1.7]:
            D = D_matrix(N, r)
            rel = commutator_norm(P, D)
            C_abs = np.linalg.norm(P @ D - D @ P)
            print(f"{N:>3} {W2:>8.0f} {r:>6.2f} {C_abs:>14.4f} {rel:>10.4f}")
        print()
