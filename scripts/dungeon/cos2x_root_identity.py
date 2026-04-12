"""
Wurzel-Identitaet: cos^2 - sin^2 = cos(2x) im Shift-Overlap-Kontext.

Ziel: Zeige dass D_nm(r) = <cos_n, S_r cos_m> - <sin_{n+1}, S_r sin_{m+1}>
sich auf eine einzige Fouriermode reduziert.

Algebra:
  cos(a)cos(b) - sin(a')sin(b') mit a=k_n*t, b=k_m*(t-s), a'=k_{n+1}*t, b'=k_{m+1}*(t-s)
  Produkt-zu-Summe:
    cos(a)cos(b) = 1/2 [cos(a-b) + cos(a+b)]
    sin(a')sin(b') = 1/2 [cos(a'-b') - cos(a'+b')]
  Differenz:
    = 1/2 [cos(a-b) - cos(a'-b') + cos(a+b) + cos(a'+b')]

Im Allgemeinen HAT das vier Terme, nicht einen. Aber wenn wir die Basis
geschickt waehlen (n -> n+1 shift), kanzelliert sich ein Teil.

Numerischer Test: Wie stark ist D_nm(r) - [cos(sum)-Anteil allein]?
"""

import numpy as np
from scipy.integrate import quad

L = 1.0

def cos_basis(n, t):
    """Normierter cos-Basisvektor auf [-L,L]. e_0 = 1/sqrt(2L), e_n = cos(n*pi*t/L)/sqrt(L)."""
    if n == 0:
        return 1.0 / np.sqrt(2*L)
    return np.cos(n * np.pi * t / L) / np.sqrt(L)

def sin_basis(n, t):
    """Normierter sin-Basisvektor: f_n = sin((n+1)*pi*t/L)/sqrt(L), n >= 0."""
    return np.sin((n+1) * np.pi * t / L) / np.sqrt(L)

def overlap_cos(n, m, s):
    """<e_n, S_s e_m> = integral cos(n*pi*t/L)*cos(m*pi*(t-s)/L) / L dt auf Ueberlap."""
    # Ueberlappungsbereich: t in [-L, L] UND t-s in [-L, L] => t in [max(-L, s-L), min(L, s+L)]
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    norm_n = 1.0/np.sqrt(2*L) if n == 0 else 1.0/np.sqrt(L)
    norm_m = 1.0/np.sqrt(2*L) if m == 0 else 1.0/np.sqrt(L)
    def integrand(t):
        if n == 0:
            fn = 1.0
        else:
            fn = np.cos(n * np.pi * t / L)
        if m == 0:
            fm = 1.0
        else:
            fm = np.cos(m * np.pi * (t - s) / L)
        return fn * fm
    val, _ = quad(integrand, a, b, limit=200)
    return val * norm_n * norm_m

def overlap_sin(n, m, s):
    """<f_n, S_s f_m>, f_n = sin((n+1)*pi*t/L)/sqrt(L)."""
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    def integrand(t):
        return np.sin((n+1)*np.pi*t/L) * np.sin((m+1)*np.pi*(t-s)/L)
    val, _ = quad(integrand, a, b, limit=200)
    return val / L

def D_nm(n, m, s):
    """Differenzmatrix-Eintrag fuer gerade n,m."""
    return overlap_cos(n, m, s) - overlap_sin(n, m, s)

# Test: zentrale Einsicht - Produkt-zu-Summe-Zerlegung
# cos(a)*cos(b) - sin(a')*sin(b')
# wobei a = n*pi*t/L, b = m*pi*(t-s)/L, a' = (n+1)*pi*t/L, b' = (m+1)*pi*(t-s)/L
#
# cos(a)cos(b) = 1/2 [cos((n-m)*pi*t/L + m*pi*s/L) + cos((n+m)*pi*t/L - m*pi*s/L)]
# sin(a')sin(b') = 1/2 [cos((n-m)*pi*t/L + (m+1)*pi*s/L - 0) - cos((n+m+2)*pi*t/L - (m+1)*pi*s/L)]
# Achtung: a' - b' = (n+1)*pi*t/L - (m+1)*pi*(t-s)/L = (n-m)*pi*t/L + (m+1)*pi*s/L
# a' + b' = (n+m+2)*pi*t/L - (m+1)*pi*s/L

print("=== Test der Wurzelidentitaet fuer n, m klein ===")
print(f"{'n':>2} {'m':>2} {'s':>6} {'D_cos':>10} {'D_sin':>10} {'Diff':>10}")
print("-" * 50)

import itertools
s_vals = [0.3, 0.5, 0.8, 1.0, 1.3, 1.5]
for (n, m) in [(0,0), (1,1), (0,1), (2,2), (1,2), (0,2), (3,3)]:
    for s in s_vals:
        dc = overlap_cos(n, m, s)
        ds = overlap_sin(n, m, s)
        print(f"{n:>2} {m:>2} {s:>6.2f} {dc:>10.5f} {ds:>10.5f} {dc-ds:>10.5f}")
    print()
