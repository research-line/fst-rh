# prolate_prime_verification_exact_kernel.py
# Requirements: Python 3.9+, numpy, scipy, mpmath, tqdm
# mp.mp.dps controls precision (decimal digits). Use mp.iv for interval mode if available.

import numpy as np
import math
from math import log
import mpmath as mp
from scipy.linalg import eigh
from tqdm import tqdm
from scipy.interpolate import interp1d
import csv

mp.mp.dps = 80  # set precision (increase for more rigorous runs)

# ---------------- Model parameters (adjustable) ----------------
LAMBDA_LIST = [100, 200]   # test values
N = 120                    # number of PSWF modes (truncation)
QUAD_MULT = 3              # quad_n = QUAD_MULT * N
P_MAX = 2000               # primes up to P_MAX (increase later)
MAX_M = 8                  # shifts m in [1..MAX_M]
EPS_LIST = [None, 0.5, 0.2, 0.1, 0.05]  # None = unregularized, others = Gaussian eps
# ----------------------------------------------------------------

def primes_upto(n):
    sieve = np.ones(n+1, dtype=bool)
    sieve[:2] = False
    for i in range(2, int(n**0.5)+1):
        if sieve[i]:
            sieve[i*i:n+1:i] = False
    return np.nonzero(sieve)[0].tolist()

def legendre_nodes_weights(n):
    from numpy.polynomial.legendre import leggauss
    x, w = leggauss(n)
    return x, w

def rescale(x, a, b):
    return 0.5*(b-a)*x + 0.5*(b+a)

# ---------------- Prolate (PSWF) discretization ----------------
def compute_prolate_modes(L, lam, N_modes, quad_n=None):
    if quad_n is None:
        quad_n = QUAD_MULT * N_modes
    x0, w0 = legendre_nodes_weights(quad_n)
    nodes = rescale(x0, -L, L)
    weights = 0.5*(L - (-L)) * w0
    sqrt_lam = math.sqrt(lam)
    def kernel(t, u):
        diff = t - u
        if abs(diff) < 1e-18:
            return sqrt_lam / math.pi
        return math.sin(sqrt_lam * diff) / (math.pi * diff)
    M = np.zeros((quad_n, quad_n), dtype=float)
    for i, ti in enumerate(nodes):
        for j, uj in enumerate(nodes):
            M[i, j] = kernel(ti, uj) * weights[j]
    vals, vecs = eigh(M)
    idx = np.argsort(vals)[::-1][:N_modes]
    vals = vals[idx]
    vecs = vecs[:, idx]
    for k in range(vecs.shape[1]):
        norm_sq = np.sum((vecs[:, k]**2) * weights)
        vecs[:, k] = vecs[:, k] / math.sqrt(norm_sq)
    return vals, vecs, nodes, weights

# ---------------- exact prime-shift kernel contribution ----------------
def weight_w(p, m):
    return (math.log(p) / (p**(m/2.0)))

def compute_M_for_p_exact(p, lam, psi_samples, nodes, weights, max_m=8, epsilon=None):
    nodes_arr = nodes
    N_modes = psi_samples.shape[1]
    M = np.zeros((N_modes, N_modes), dtype=complex)
    # For each m>=1 include both +m and -m shifts with weight log(p)/p^{m/2}
    for m in range(1, max_m+1):
        wpm = weight_w(p, m)
        shift = m * math.log(p)
        for sign in [+1, -1]:
            s = sign * shift
            if epsilon is None:
                # discrete translation via cubic interpolation (zero outside [-L,L])
                for n in range(N_modes):
                    f_u = psi_samples[:, n]
                    interp = interp1d(nodes_arr, f_u, kind='cubic', fill_value=0.0, bounds_error=False)
                    translated = interp(nodes_arr + s)
                    for mm in range(N_modes):
                        M[mm, n] += wpm * np.sum(np.conjugate(psi_samples[:, mm]) * translated * weights)
            else:
                fac = 1.0 / (epsilon * math.sqrt(math.pi))
                # build kernel matrix K_ij = fac * exp(-((t_i - u_j - s)/eps)^2)
                K = np.empty((len(nodes_arr), len(nodes_arr)), dtype=float)
                for i, ti in enumerate(nodes_arr):
                    for j, uj in enumerate(nodes_arr):
                        K[i, j] = fac * math.exp(-((ti - uj - s)/epsilon)**2)
                for n in range(N_modes):
                    v = K.dot(psi_samples[:, n] * weights)
                    for mm in range(N_modes):
                        M[mm, n] += wpm * np.sum(np.conjugate(psi_samples[:, mm]) * v)
    return M

# ---------------- main verification loop ----------------
def run_verification_exact(LAMBDA_LIST, N, P_MAX, MAX_M, EPS_LIST):
    primes = primes_upto(P_MAX)
    results = {}
    for lam in LAMBDA_LIST:
        L = math.log(lam)
        vals, vecs, nodes, weights = compute_prolate_modes(L, lam, N)
        psi_samples = vecs
        even_idx = list(range(0, N, 2))
        odd_idx = list(range(1, N, 2))
        sum_Delta = mp.mpf('0')
        sum_HS_off = mp.mpf('0')
        Delta_list = []
        HS_list = []
        max_off_abs = 0.0
        count_pos = 0
        for p in tqdm(primes, desc=f"lambda={lam} primes"):
            M = compute_M_for_p_exact(p, lam, psi_samples, nodes, weights, max_m=MAX_M, epsilon=None)
            Sigma = np.real(np.diag(M))
            Delta_p = float(np.min(Sigma[even_idx]) - np.max(Sigma[odd_idx]))
            off_sq = 0.0
            for i in range(N):
                for j in range(N):
                    if i != j:
                        off_sq += abs(M[i, j])**2
                        if abs(M[i,j]) > max_off_abs:
                            max_off_abs = abs(M[i,j])
            HS_off = math.sqrt(off_sq)
            Delta_list.append((p, Delta_p))
            HS_list.append((p, HS_off))
            sum_Delta += mp.mpf(str(Delta_p))
            sum_HS_off += mp.mpf(str(HS_off))
            if Delta_p > 0:
                count_pos += 1
        gap_est = sum_Delta - sum_HS_off
        top10 = sorted(Delta_list, key=lambda x: -x[1])[:10]
        results[lam] = {
            'sum_Delta': sum_Delta,
            'sum_HS_off': sum_HS_off,
            'gap_estimate': gap_est,
            'top10': top10,
            'max_off': max_off_abs,
            'count_positive_Delta': count_pos,
            'Delta_list': Delta_list,
            'HS_list': HS_list
        }
    return results

if __name__ == "__main__":
    res = run_verification_exact(LAMBDA_LIST, N, P_MAX, MAX_M, EPS_LIST)
    for lam, r in res.items():
        print("lambda", lam, "sum_Delta", r['sum_Delta'], "sum_HS_off", r['sum_HS_off'], "gap", r['gap_estimate'])
        
        with open(f'results_lambda_{lam}.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['p', 'Delta_p', 'HS_off'])
            for (p, d), (p2, h) in zip(r['Delta_list'], r['HS_list']):
                writer.writerow([p, d, h])
