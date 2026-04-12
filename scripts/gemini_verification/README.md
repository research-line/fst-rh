# Gemini Prolate Prime Verification

Session: 2026-04-12 (Gemini workstation), transferred to project 2026-04-13.

## Purpose

Independent verification of the Shift Parity Lemma's consequences in the full
Galerkin space (N=120 PSWF modes) at $\lambda=100$ and $\lambda=200$, with
primes summed up to $P_{\max} \approx 2000$.

## Files

### Initial Run (2026-04-12, P_max = 2000)
- `prolate_prime_verification_exact_kernel.py` — The verification script v1
  (PSWF basis, exact sin(√λ·diff)/(π·diff) kernel, interval precision via mpmath)
- `results_lambda_100.csv` — 303 primes, column `Delta_p` is the per-prime
  contribution to the even–odd gap
- `results_lambda_200.csv` — 303 primes, same format

### Production Run (2026-04-12 02:52 UTC, P_max = 10000)
- `prolate_prime_verification_exact_kernel_v2.py` — updated script (N=120, P_max=10000)
- `results_lambda_100_N120_P1229.csv` — 1229 primes at λ=100
- `results_lambda_200_N120_P1229.csv` — 1229 primes at λ=200

## Key result (Production Run)

At both $\lambda=100$ and $\lambda=200$ with N=120 Galerkin modes and primes up to $P_\max = 10{,}000$:
- **1229/1229 primes** have $\Delta_p < 0$ (every prime deepens the gap)
- **Zero exceptions** across 2458 individual prime tests

This confirms the Shift Parity Lemma's cumulative behavior in the full Galerkin
space: no prime prefers the odd sector, and the p=2 anomaly observed in the 3×3
truncation vanishes when enough modes are included. The production run covers
nearly one order of magnitude more primes than the initial run (1229 vs 303)
with no new exceptions found.

## How to reproduce

Dependencies: numpy, scipy, mpmath, tqdm.

```bash
PYTHONIOENCODING=utf-8 python prolate_prime_verification_exact_kernel.py
```

Runtime: ~2–3 hours on CCX13-class hardware (2 vCPU, 8 GB). The quadrature grid
is $3 \times N = 360$ points; eigenvalue computation is $O(N^3)$.

## Parameters

- `N = 120` PSWF modes
- `QUAD_MULT = 3` (quadrature nodes = $3N$)
- `P_MAX = 2000`
- `MAX_M = 8` (shift orders)
- `mp.mp.dps = 80` (decimal precision)

## Credit

Script authored by Gemini (Google DeepMind, assisting the author's research).
Transferred to the project via the cross-system sync folder for archival
alongside the main proof artifacts.
