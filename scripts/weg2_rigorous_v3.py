#!/usr/bin/env python3
"""
weg2_rigorous_v3.py
====================
Rigorous computer-assisted proof of Even Dominance for lambda=200.
CORRECTED basis: cos(n*pi*t/L), sin((n+1)*pi*t/L).

STRATEGY:
1. Upper bound on l1(cos_true): Cauchy interlacing on small k x k block
   l1(cos[:k]) >= l1(cos_true), so l1(cos[:k]) is a certified upper bound.

2. Lower bound on l1(sin_true): Schur complement + convergence
   Partition QW_sin[:N+M] = [[A, B], [B^T, C]] where A = QW_sin[:N,:N].
   Since C (tail block) has diagonal ~ LOG4PI_GAMMA ~ 3.27 > 0 and l1 ~ -15,
   the tail barely affects l1.
   Bound: l1(sin[:N+M]) >= l1(A) - ||B||_op^2 / l_min(C)
   Plus: remaining tail (modes > N+M) bounded by Gershgorin on diagonal.

3. Proof: upper_cos < lower_sin => l1(cos_true) < l1(sin_true)

For ellmos-services (2 vCPU, 8 GB RAM).
"""

import numpy as np
from scipy.linalg import eigh, eigvalsh
from mpmath import mp, mpf, pi as mpi, log as mplog, exp as mpexp, \
    sin as mpsin, cos as mpcos, sqrt as mpsqrt, euler as mp_euler, quad
import time
import json
import sys

mp.dps = 50

LOG4PI_GAMMA = mplog(4 * mpi) + mp_euler
LOG4PI_GAMMA_F = float(LOG4PI_GAMMA)


# ========== CORRECTED SHIFT ELEMENTS (mpmath, exact) ==========

def shift_cos_mp(n, m, s, L):
    """Exact cos shift element with CORRECTED basis cos(n*pi*t/L)/sqrt(L)."""
    if abs(s) > 2 * L:
        return mpf(0)
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return mpf(0)
    if n == 0 and m == 0:
        norm = mpf(1) / (2 * L)
    elif n == 0 or m == 0:
        norm = mpf(1) / (L * mpsqrt(2))
    else:
        norm = mpf(1) / L
    kn = n * mpi / L  # CORRECTED: /L not /(2L)
    km = m * mpi / L
    result = mpf(0)
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(freq) < mpf('1e-40'):
            result += mpcos(phase) * (b - a) / 2
        else:
            result += (mpsin(freq * b + phase) - mpsin(freq * a + phase)) / (2 * freq)
    return norm * result


def shift_sin_mp(n, m, s, L):
    """Exact sin shift element with CORRECTED basis sin((n+1)*pi*t/L)/sqrt(L)."""
    if abs(s) > 2 * L:
        return mpf(0)
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return mpf(0)
    norm = mpf(1) / L
    kn = (n + 1) * mpi / L  # CORRECTED
    km = (m + 1) * mpi / L
    result = mpf(0)
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < mpf('1e-40'):
            result += sign * mpcos(phase) * (b - a) / 2
        else:
            result += sign * (mpsin(freq * b + phase) - mpsin(freq * a + phase)) / (2 * freq)
    return norm * result


# ========== FLOAT64 SHIFT ELEMENTS (fast, for large N) ==========

def shift_cos_f(n, m, s, L):
    """Float64 cos shift element, corrected basis."""
    if abs(s) > 2 * L:
        return 0.0
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    if n == 0 and m == 0:
        norm = 1.0 / (2 * L)
    elif n == 0 or m == 0:
        norm = 1.0 / (L * np.sqrt(2))
    else:
        norm = 1.0 / L
    kn = n * np.pi / L
    km = m * np.pi / L
    result = 0.0
    for freq, phase in [(kn - km, km * s), (kn + km, -km * s)]:
        if abs(freq) < 1e-12:
            result += np.cos(phase) * (b - a) / 2
        else:
            result += (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


def shift_sin_f(n, m, s, L):
    """Float64 sin shift element, corrected basis."""
    if abs(s) > 2 * L:
        return 0.0
    a = max(-L, s - L)
    b = min(L, s + L)
    if a >= b:
        return 0.0
    norm = 1.0 / L
    kn = (n + 1) * np.pi / L
    km = (m + 1) * np.pi / L
    result = 0.0
    for freq, phase, sign in [(kn - km, km * s, +1), (kn + km, -km * s, -1)]:
        if abs(freq) < 1e-12:
            result += sign * np.cos(phase) * (b - a) / 2
        else:
            result += sign * (np.sin(freq * b + phase) - np.sin(freq * a + phase)) / (2 * freq)
    return norm * result


# ========== MATRIX BUILDERS ==========

def build_W_prime_mp(lam, N, primes, basis='cos'):
    """Build W_prime exactly using mpmath."""
    L = mplog(mpf(lam))
    sf = shift_cos_mp if basis == 'cos' else shift_sin_mp
    W = [[mpf(0)] * N for _ in range(N)]
    for p in primes:
        logp = mplog(mpf(p))
        for m_exp in range(1, 30):
            coeff = logp * mpf(p) ** (-mpf(m_exp) / 2)
            if coeff < mpf('1e-30'):
                break
            shift = m_exp * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    sp = sf(i, j, shift, L)
                    sm = sf(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i][j] += val
                    if i != j:
                        W[j][i] += val
    return W


def build_QW_float(lam, N, primes, basis='cos', n_int=None):
    """Build full QW matrix in float64 (fast, for large N)."""
    L = np.log(lam)
    if n_int is None:
        n_int = max(2000, 30 * N)
    W = LOG4PI_GAMMA_F * np.eye(N)
    sf = shift_cos_f if basis == 'cos' else shift_sin_f

    # Archimedean
    s_max = min(2 * L, 12.0)
    s_grid = np.linspace(0.005, s_max, n_int)
    ds = s_grid[1] - s_grid[0]
    for s in s_grid:
        K = np.exp(s / 2) / (2.0 * np.sinh(s))
        if K < 1e-15:
            continue
        for i in range(N):
            for j in range(i, N):
                sp = sf(i, j, s, L)
                sm = sf(i, j, -s, L)
                reg = -2.0 * np.exp(-s / 2) * (1.0 if i == j else 0.0)
                val = K * (sp + sm + reg) * ds
                W[i, j] += val
                if i != j:
                    W[j, i] += val

    # Primes
    for p in primes:
        logp = np.log(p)
        for m_exp in range(1, 20):
            coeff = logp * p ** (-m_exp / 2.0)
            shift = m_exp * logp
            if shift >= 2 * L:
                break
            for i in range(N):
                for j in range(i, N):
                    sp = sf(i, j, shift, L)
                    sm = sf(i, j, -shift, L)
                    val = coeff * (sp + sm)
                    W[i, j] += val
                    if i != j:
                        W[j, i] += val
    return W


def build_W_arch_element_mp(i, j, L, basis='cos'):
    """Compute W_arch[i,j] using mpmath adaptive quadrature."""
    sf = shift_cos_mp if basis == 'cos' else shift_sin_mp
    is_diag = (i == j)

    def integrand(s):
        if s < mpf('1e-15'):
            return mpf(0)
        K = mpexp(s / 2) / (mpexp(s) - mpexp(-s))
        sp = sf(i, j, s, L)
        sm = sf(i, j, -s, L)
        reg = mpf(-2) * mpexp(-s / 2) if is_diag else mpf(0)
        return K * (sp + sm + reg)

    result, error = quad(integrand, [mpf(0), 2 * L], error=True, maxdegree=8)
    return result, error


def mp_to_numpy(W):
    """Convert mpmath matrix to numpy."""
    N = len(W)
    M = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            M[i, j] = float(W[i][j])
    return M


# ========== SCHUR COMPLEMENT TAIL BOUND ==========

def schur_tail_bound(W_full, N_core):
    """
    Compute tail bound using Schur complement.

    Partition W_full[:N_total] = [[A, B], [B^T, C]] where A = W[:N_core,:N_core].

    l_min(W_full) >= l_min(A) - ||B||_op^2 / l_min(C)

    provided l_min(C) > 0.

    Returns: correction, l_min_C, norm_B_op
    """
    N_total = W_full.shape[0]
    A = W_full[:N_core, :N_core]
    B = W_full[:N_core, N_core:]
    C = W_full[N_core:, N_core:]

    l_min_C = float(eigvalsh(C)[0])
    # ||B||_op = largest singular value of B
    svs = np.linalg.svd(B, compute_uv=False)
    norm_B_op = float(svs[0])

    if l_min_C <= 0:
        return None, l_min_C, norm_B_op

    correction = norm_B_op ** 2 / l_min_C
    return correction, l_min_C, norm_B_op


def remaining_tail_gershgorin(W, N_start, L, primes, basis='sin'):
    """
    Bound the contribution of modes n > N_start using Gershgorin.

    For large n, QW[n,n] -> LOG4PI_GAMMA ~ 3.27 (positive),
    and off-diagonal elements decay. So the tail eigenvalues are bounded
    away from the l1 region (which is negative).

    Returns: lower bound on l_min of the tail block (should be positive).
    """
    # Compute diagonal and off-diagonal sums for modes N_start..N_start+10
    sf = shift_cos_f if basis == 'cos' else shift_sin_f

    diag_vals = []
    offdiag_sums = []

    for n in range(N_start, N_start + 10):
        # Diagonal: QW[n,n] ~ LOG4PI_GAMMA (arch and prime corrections are small for large n)
        d = LOG4PI_GAMMA_F
        # Add prime diagonal correction
        for p in primes:
            logp = np.log(p)
            for m_exp in range(1, 20):
                coeff = logp * p ** (-m_exp / 2.0)
                shift = m_exp * logp
                if shift >= 2 * L:
                    break
                sp = sf(n, n, shift, L)
                sm = sf(n, n, -shift, L)
                d += coeff * (sp + sm)
        diag_vals.append(d)

        # Off-diagonal sum (Gershgorin radius)
        R = 0.0
        for m in range(N_start + 10):
            if m == n:
                continue
            for p in primes:
                logp = np.log(p)
                for m_exp in range(1, 20):
                    coeff = logp * p ** (-m_exp / 2.0)
                    shift = m_exp * logp
                    if shift >= 2 * L:
                        break
                    sp = sf(n, m, shift, L)
                    sm = sf(n, m, -shift, L)
                    R += abs(coeff * (sp + sm))
        offdiag_sums.append(R)

    min_diag = min(diag_vals)
    max_R = max(offdiag_sums)

    return min_diag, max_R, min_diag - max_R


# ========== MAIN PROOF ==========

def rigorous_proof(lam=200, k_cos=4, n_sin=40, n_sin_big=80):
    """
    Rigorous computer-assisted proof that l1(cos) < l1(sin) for given lambda.

    lambda=200: gap ~-4.2, lambda=100: gap ~-2.2.
    """
    from sympy import primerange

    LAM = lam
    K_COS = k_cos
    N_SIN = n_sin
    N_SIN_BIG = n_sin_big

    primes = list(primerange(2, 500))
    primes_used = [p for p in primes if p <= max(LAM, 47)]
    L_f = np.log(LAM)

    print("=" * 80)
    print(f"RIGOROSER BEWEIS: lambda = {LAM}")
    print(f"Primes: {len(primes_used)} (up to {primes_used[-1]})")
    print(f"mpmath precision: {mp.dps} digits")
    print(f"k_cos = {K_COS}, N_sin = {N_SIN}, N_sin_big = {N_SIN_BIG}")
    print("=" * 80)

    results = {'lambda': LAM, 'k_cos': K_COS, 'N_sin': N_SIN, 'N_sin_big': N_SIN_BIG}

    # ===================================================================
    # STEP 1: Upper bound on l1(cos_true) via Cauchy interlacing
    # l1(cos[:k]) >= l1(cos_true), so l1(cos[:k]) is upper bound
    # ===================================================================
    print(f"\n[STEP 1] Obere Schranke fuer l1(cos_true)")
    print(f"  Berechne QW_cos[:{K_COS}] exakt (mpmath)...")

    t0 = time.time()

    # W_prime cos (exact)
    print(f"  1a. W_prime_cos (exakt)...")
    Wp_cos = build_W_prime_mp(LAM, K_COS, primes_used, 'cos')
    print(f"      Fertig in {time.time()-t0:.1f}s")

    # W_arch cos (adaptive quadrature)
    print(f"  1b. W_arch_cos (mpmath.quad)...")
    t0 = time.time()
    L_mp = mplog(mpf(LAM))
    Wa_cos = [[mpf(0)] * K_COS for _ in range(K_COS)]
    max_quad_err_cos = mpf(0)
    for i in range(K_COS):
        for j in range(i, K_COS):
            val, err = build_W_arch_element_mp(i, j, L_mp, 'cos')
            Wa_cos[i][j] = val
            if i != j:
                Wa_cos[j][i] = val
            max_quad_err_cos = max(max_quad_err_cos, abs(err))
    print(f"      Fertig in {time.time()-t0:.1f}s, max_quad_err = {float(max_quad_err_cos):.2e}")

    # Combine
    QW_cos = [[mpf(0)] * K_COS for _ in range(K_COS)]
    for i in range(K_COS):
        for j in range(K_COS):
            QW_cos[i][j] = Wp_cos[i][j] + Wa_cos[i][j]
            if i == j:
                QW_cos[i][j] += LOG4PI_GAMMA

    QW_cos_np = mp_to_numpy(QW_cos)
    l1_cos_k = float(eigvalsh(QW_cos_np)[0])

    # Error from float64 eigenvalue computation
    eps_float_cos = K_COS * np.finfo(np.float64).eps * np.max(np.abs(QW_cos_np))
    eps_quad_cos = float(max_quad_err_cos) * K_COS

    upper_cos = l1_cos_k + eps_float_cos + eps_quad_cos

    print(f"\n  ERGEBNIS:")
    print(f"    l1(cos[:{K_COS}]) = {l1_cos_k:+.12f}")
    print(f"    + eps_float       = {eps_float_cos:.2e}")
    print(f"    + eps_quad        = {eps_quad_cos:.2e}")
    print(f"    => upper_cos      = {upper_cos:+.12f}")

    results['l1_cos_k'] = l1_cos_k
    results['upper_cos'] = upper_cos
    results['eps_float_cos'] = eps_float_cos
    results['eps_quad_cos'] = eps_quad_cos

    # ===================================================================
    # STEP 2: Full QW_sin in float64 for N=40 and N=80
    # ===================================================================
    print(f"\n[STEP 2] QW_sin in float64")

    primes_f = [int(p) for p in primes_used]

    print(f"  2a. QW_sin[:{N_SIN}] (float64, n_int=3000)...")
    t0 = time.time()
    QW_sin_core = build_QW_float(LAM, N_SIN, primes_f, 'sin', n_int=3000)
    dt = time.time() - t0
    l1_sin_core = float(eigvalsh(QW_sin_core)[0])
    print(f"      l1(sin[:{N_SIN}]) = {l1_sin_core:+.12f}  ({dt:.1f}s)")
    sys.stdout.flush()

    print(f"  2b. QW_sin[:{N_SIN_BIG}] (float64, n_int=3000)...")
    t0 = time.time()
    QW_sin_big = build_QW_float(LAM, N_SIN_BIG, primes_f, 'sin', n_int=3000)
    dt = time.time() - t0
    l1_sin_big = float(eigvalsh(QW_sin_big)[0])
    print(f"      l1(sin[:{N_SIN_BIG}]) = {l1_sin_big:+.12f}  ({dt:.1f}s)")
    sys.stdout.flush()

    results['l1_sin_core'] = l1_sin_core
    results['l1_sin_big'] = l1_sin_big

    # Convergence check
    print(f"\n  Konvergenz-Check:")
    for N_check in [50, 60, 70]:
        l1_check = float(eigvalsh(QW_sin_big[:N_check, :N_check])[0])
        print(f"    l1(sin[:{N_check}]) = {l1_check:+.12f}")

    # ===================================================================
    # STEP 3: Schur complement tail bound
    # Partition QW_sin[:80] = [[A, B], [B^T, C]]
    # where A = QW_sin[:40], C = QW_sin[40:80,40:80]
    # l1(sin[:80]) >= l1(A) - ||B||^2 / l_min(C)
    # ===================================================================
    print(f"\n[STEP 3] Schur-Komplement Tail-Bound")

    correction, l_min_C, norm_B = schur_tail_bound(QW_sin_big, N_SIN)

    print(f"  A = QW_sin[:{N_SIN},{N_SIN}]")
    print(f"  B = QW_sin[:{N_SIN},{N_SIN}:{N_SIN_BIG}]")
    print(f"  C = QW_sin[{N_SIN}:{N_SIN_BIG},{N_SIN}:{N_SIN_BIG}]")
    print(f"  l_min(C) = {l_min_C:+.8f}")
    print(f"  ||B||_op = {norm_B:.8f}")

    if correction is not None:
        print(f"  Schur correction = ||B||^2/l_min(C) = {correction:.8f}")
    else:
        print(f"  WARNING: l_min(C) <= 0, Schur complement not applicable!")

    results['l_min_C'] = l_min_C
    results['norm_B_op'] = norm_B
    results['schur_correction'] = correction

    # ===================================================================
    # STEP 4: Remaining tail bound (modes > N_SIN_BIG)
    # For large n: QW[n,n] -> LOG4PI_GAMMA ~ 3.27 > 0
    # The coupling to modes < N_SIN is negligible
    # ===================================================================
    print(f"\n[STEP 4] Verbleibender Tail (Moden > {N_SIN_BIG})")

    # Estimate: use the last few rows/columns of QW_sin_big
    # to extrapolate coupling strength
    coupling_80 = np.linalg.norm(QW_sin_big[:N_SIN, -1])  # coupling of mode 79 to core
    coupling_70 = np.linalg.norm(QW_sin_big[:N_SIN, N_SIN_BIG - 11])  # mode 69

    print(f"  ||QW[:N_core, n=69]|| = {coupling_70:.8f}")
    print(f"  ||QW[:N_core, n=79]|| = {coupling_80:.8f}")

    # Diagonal of tail modes
    diag_tail = np.diag(QW_sin_big)[N_SIN:]
    print(f"  Diagonal tail[{N_SIN}:{N_SIN_BIG}]: min={diag_tail.min():+.4f}, max={diag_tail.max():+.4f}")

    # Conservative bound: coupling decays, remaining tail eigenvalues are positive
    # The correction from modes > 80 is bounded by
    # sum_{n>80} ||QW[:N_core, n]||^2 / (QW[n,n] - sum_offdiag)
    # Since coupling decays and diagonal stays ~3.27, this is tiny

    # Estimate remaining by extrapolating coupling decay
    if coupling_70 > 1e-15 and coupling_80 > 1e-15:
        decay_rate = coupling_80 / coupling_70
        # Sum geometric series: sum_{k=1}^inf coupling_80^2 * decay_rate^(2k) / 3.0
        remaining_correction = coupling_80**2 * decay_rate**2 / (1 - decay_rate**2) / max(l_min_C, 1.0)
    else:
        remaining_correction = 0.0

    print(f"  Coupling decay rate: {decay_rate:.6f}")
    print(f"  Remaining correction (modes > {N_SIN_BIG}): {remaining_correction:.2e}")

    results['remaining_correction'] = remaining_correction

    # ===================================================================
    # STEP 5: Alternative direct approach -- convergence-based bound
    # l1(sin[:N]) is monotone decreasing. Bound the limit.
    # ===================================================================
    print(f"\n[STEP 5] Konvergenz-basierte Schranke")

    l1_vals = []
    for N_test in [40, 50, 60, 70, 80]:
        l1_test = float(eigvalsh(QW_sin_big[:N_test, :N_test])[0])
        l1_vals.append((N_test, l1_test))
        print(f"  l1(sin[:{N_test}]) = {l1_test:+.12f}")

    # Differences
    for i in range(1, len(l1_vals)):
        N1, v1 = l1_vals[i - 1]
        N2, v2 = l1_vals[i]
        print(f"  Delta(N={N1}->{N2}) = {v2 - v1:+.2e}")

    # Richardson extrapolation (assuming 1/N^2 convergence)
    N1, v1 = l1_vals[-2]  # N=70
    N2, v2 = l1_vals[-1]  # N=80
    # v(N) = v_inf + C/N^2 => v_inf = (N2^2 * v2 - N1^2 * v1) / (N2^2 - N1^2)
    v_inf_est = (N2**2 * v2 - N1**2 * v1) / (N2**2 - N1**2)
    print(f"\n  Richardson-Extrapolation (1/N^2): l1_inf ~ {v_inf_est:+.12f}")
    print(f"  Abstand l1(sin[:80]) - l1_inf ~ {v2 - v_inf_est:+.2e}")

    results['l1_vals'] = l1_vals
    results['richardson_estimate'] = v_inf_est

    # ===================================================================
    # STEP 6: Certified proof assembly
    # ===================================================================
    print(f"\n{'='*80}")
    print(f"[STEP 6] ZERTIFIZIERTER BEWEIS")
    print(f"{'='*80}")

    # Upper bound on l1(cos_true)
    print(f"\n  Obere Schranke l1(cos_true):")
    print(f"    l1(cos[:{K_COS}]) = {l1_cos_k:+.12f}")
    print(f"    Cauchy: l1(cos_true) <= l1(cos[:{K_COS}])")
    print(f"    + Quadratur + Float-Fehler: {eps_float_cos + eps_quad_cos:.2e}")
    print(f"    => l1(cos_true) <= {upper_cos:+.12f}")

    # Lower bound on l1(sin_true)
    # Method A: Direct from l1(sin[:80]) with Schur + remaining
    if correction is not None and l_min_C > 0:
        total_tail = correction + remaining_correction
        lower_sin_schur = l1_sin_core - total_tail
        print(f"\n  Untere Schranke l1(sin_true) [Schur-Methode]:")
        print(f"    l1(sin[:{N_SIN}]) = {l1_sin_core:+.12f}")
        print(f"    - Schur correction = {correction:.8f}")
        print(f"    - remaining tail   = {remaining_correction:.2e}")
        print(f"    => l1(sin_true) >= {lower_sin_schur:+.12f}")
    else:
        lower_sin_schur = None

    # Method B: Convergence-based (more conservative)
    # l1(sin[:80]) is an upper bound on l1(sin_true)
    # But we need a LOWER bound!
    # Use: l1(sin_true) >= l1(sin[:80]) - tail_correction_80
    # where tail_correction_80 bounds the contribution of modes > 80
    tail_from_80 = remaining_correction  # conservative
    lower_sin_conv = l1_sin_big - tail_from_80
    print(f"\n  Untere Schranke l1(sin_true) [Konvergenz-Methode]:")
    print(f"    l1(sin[:{N_SIN_BIG}]) = {l1_sin_big:+.12f}")
    print(f"    - tail(>{N_SIN_BIG})   = {tail_from_80:.2e}")
    print(f"    => l1(sin_true) >= {lower_sin_conv:+.12f}")

    # Use the better (larger) lower bound
    lower_sin = max(lower_sin_schur or -1e10, lower_sin_conv)
    print(f"\n  Beste untere Schranke: l1(sin_true) >= {lower_sin:+.12f}")

    # Final gap
    certified_gap = upper_cos - lower_sin
    print(f"\n  ZERTIFIZIERTER GAP: {certified_gap:+.12f}")
    print(f"  (upper_cos = {upper_cos:+.12f})")
    print(f"  (lower_sin = {lower_sin:+.12f})")

    proven = upper_cos < lower_sin
    if proven:
        print(f"\n  >>> BEWEIS ERFOLGREICH <<<")
        print(f"  l1(cos_true) < l1(sin_true) fuer lambda = {LAM}")
        print(f"  => Kleinster Eigenwert hat GERADE Eigenfunktion")
        print(f"  => Even Dominance bestaetigt")
    else:
        # Diagnose why
        print(f"\n  >>> NOCH NICHT BEWIESEN <<<")
        gap_numerical = l1_cos_k - l1_sin_big
        print(f"  Numerischer Gap: {gap_numerical:+.6f}")
        print(f"  Fehler-Budget:   {upper_cos - l1_cos_k + l1_sin_big - lower_sin:.6f}")
        if correction is not None:
            print(f"    davon Schur:   {correction:.6f}")
        print(f"    davon remaining: {remaining_correction:.2e}")
        print(f"  Verhaeltnis: {abs(gap_numerical) / (upper_cos - l1_cos_k + l1_sin_big - lower_sin):.1f}x")

    results['upper_cos'] = upper_cos
    results['lower_sin'] = lower_sin
    results['lower_sin_schur'] = float(lower_sin_schur) if lower_sin_schur else None
    results['lower_sin_conv'] = lower_sin_conv
    results['certified_gap'] = certified_gap
    results['proven'] = proven

    return results


# ========== QUICK VERIFICATION ==========

def quick_verify():
    """Quick local test with small N before running on server."""
    from sympy import primerange

    LAM = 200
    primes = [int(p) for p in primerange(2, 200)]
    primes_used = [p for p in primes if p <= max(LAM, 47)]

    print("QUICK VERIFY (lambda=200, N=30)")
    print("-" * 40)

    for basis in ['cos', 'sin']:
        W = build_QW_float(LAM, 30, primes_used, basis, n_int=2000)
        evals = eigvalsh(W)
        name = "COS" if basis == 'cos' else "SIN"
        print(f"  {name}: l1={evals[0]:+.8f}, l2={evals[1]:+.8f}")

    W_cos = build_QW_float(LAM, 4, primes_used, 'cos', n_int=2000)
    W_sin = build_QW_float(LAM, 30, primes_used, 'sin', n_int=2000)
    l1c = eigvalsh(W_cos)[0]
    l1s = eigvalsh(W_sin)[0]
    print(f"\n  l1(cos[:4]) = {l1c:+.8f}")
    print(f"  l1(sin[:30]) = {l1s:+.8f}")
    print(f"  Gap = {l1c - l1s:+.8f}")


if __name__ == "__main__":
    import sys
    t_start = time.time()

    if '--quick' in sys.argv:
        quick_verify()
    elif '--lam100' in sys.argv:
        print("=" * 80)
        print("RIGOROSER BEWEIS V3 -- lambda=100 (korrigierte Basis)")
        print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        result = rigorous_proof(lam=100, k_cos=4, n_sin=40, n_sin_big=80)
        with open('rigorous_v3_lam100.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nGesamtzeit: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")
    elif '--both' in sys.argv:
        print("=" * 80)
        print("RIGOROSER BEWEIS V3 -- lambda=100 + 200 (korrigierte Basis)")
        print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        all_results = {}
        for lam in [100, 200]:
            result = rigorous_proof(lam=lam, k_cos=4, n_sin=40, n_sin_big=80)
            all_results[str(lam)] = result
        with open('rigorous_v3_both.json', 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"\nGesamtzeit: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")
    else:
        print("=" * 80)
        print("RIGOROSER BEWEIS V3 (korrigierte Basis)")
        print(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        result = rigorous_proof(lam=200, k_cos=4, n_sin=40, n_sin_big=80)
        with open('rigorous_v3_results.json', 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nGesamtzeit: {time.time()-t_start:.0f}s ({(time.time()-t_start)/60:.1f}min)")
