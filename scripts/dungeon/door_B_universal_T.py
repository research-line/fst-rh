"""
Tuer B (definitiv): Gibt es einen universellen symmetrischen T, der mit D(r)
fuer ALLE r in (0,2) kommutiert?

Ansatz: Sample viele r-Werte, D_i = D(r_i). Suche T symmetrisch mit [T, D_i] = 0 fuer alle i.

Parametrisiere T via seine N*(N+1)/2 unabhaengige Eintraege. Die Bedingung
[T, D_i] = 0 gibt pro r_i eine Linearform in den T-Eintraegen.

SVD der gestapelten Matrix zeigt: Kern-Dim = 1 (nur Identitaet)? Oder mehr (echte Symmetrie)?
"""

import numpy as np
from scipy.linalg import eigh
from door_B_slepian_commutator import D_matrix

def build_commutator_constraint_matrix(D_list, N):
    """
    Fuer symmetrisches T mit T_ij = t_{ij} fuer i<=j:
      [T, D]_ab = sum_k (T_ak D_kb - D_ak T_kb)
    Das ist linear in den t_{ij}. Baue die grosse Constraint-Matrix.

    Parameter-Vektor: t = [t_00, t_01, ..., t_0,N-1, t_11, t_12, ..., t_{N-1,N-1}]
    Dimension: N*(N+1)/2
    """
    num_params = N * (N + 1) // 2
    # Mapping (i,j) mit i<=j -> param-Index
    idx_map = {}
    k = 0
    for i in range(N):
        for j in range(i, N):
            idx_map[(i, j)] = k
            k += 1

    # Pro D: N*N Gleichungen (Kommutator-Eintraege)
    num_eqs = len(D_list) * N * N
    A = np.zeros((num_eqs, num_params))
    row = 0
    for D in D_list:
        for a in range(N):
            for b in range(N):
                # [T,D]_ab = sum_k (T_ak D_kb - D_ak T_kb)
                for k in range(N):
                    # T_ak: Index im Parametervektor
                    ak_i, ak_j = (a, k) if a <= k else (k, a)
                    p_ak = idx_map[(ak_i, ak_j)]
                    # T_kb:
                    kb_i, kb_j = (k, b) if k <= b else (b, k)
                    p_kb = idx_map[(kb_i, kb_j)]
                    A[row, p_ak] += D[k, b]
                    A[row, p_kb] -= D[a, k]
                row += 1
    return A, idx_map

def solve_universal_T(N, r_list, tol=1e-8):
    """Finde Kern von ||[T, D(r_i)]||^2 ueber alle i via SVD."""
    D_list = [D_matrix(N, r) for r in r_list]
    A, idx_map = build_commutator_constraint_matrix(D_list, N)
    # SVD
    U, s, Vh = np.linalg.svd(A, full_matrices=False)
    # Kern-Dimension
    kernel_dim = np.sum(s < tol * s[0])
    return s, Vh, kernel_dim, idx_map

def extract_T_from_vector(t_vec, idx_map, N):
    T = np.zeros((N, N))
    for (i, j), k in idx_map.items():
        T[i, j] = t_vec[k]
        if i != j:
            T[j, i] = t_vec[k]
    return T

print("=== Tuer B: Universeller kommutierender Operator? ===")
r_samples = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9, 1.0, 1.1, 1.25, 1.4, 1.55, 1.7, 1.85]
for N in [3, 5, 7, 10, 15]:
    s, Vh, kdim, idx_map = solve_universal_T(N, r_samples, tol=1e-6)
    num_params = N * (N + 1) // 2
    print(f"N={N:>2}: num_params={num_params:>3}, kleinste 3 sigmas={s[-3:]}, Kern-Dim (tol=1e-6)={kdim}")
    # Sigma-Relativ
    print(f"   relative Singular: sigma_min/sigma_max = {s[-1]/s[0]:.2e}")
    if kdim >= 1:
        T_opt = extract_T_from_vector(Vh[-1], idx_map, N)
        print(f"   Optimales T (nahe Kern): diag = {np.diag(T_opt)[:5]}")
        print(f"   Off-diag von T: {T_opt[0,1]:.4f}, {T_opt[1,2]:.4f}, {T_opt[2,3] if N>=4 else 0:.4f}")
    print()
