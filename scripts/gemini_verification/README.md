# Gemini Prolate Prime Verification

Session: 2026-04-12 (Gemini workstation / ellmos-services). Initial transfer:
2026-04-13. Server re-check and missing-log archival: 2026-04-30.

## Purpose

Independent verification of the Shift Parity Lemma's consequences in the full
Galerkin space at `lambda=100` and `lambda=200`.

## Files

### Initial Run (2026-04-12, N=120, P_max=2000)

- `prolate_prime_verification_exact_kernel.py` - verification script v1
  (PSWF basis, exact `sin(sqrt(lambda)*diff)/(pi*diff)` kernel).
- `results_lambda_100.csv` - 303 primes at `lambda=100`.
- `results_lambda_200.csv` - 303 primes at `lambda=200`.
- `prolate_run_N120_P2000_2026-04-11.log` - server progress log recovered on
  2026-04-30.

### Production Run (2026-04-12, N=200, P_max=10000)

- `prolate_prime_verification_exact_kernel_v2.py` - updated local copy.
- `prolate_prime_verification_exact_kernel_server_N200_P10000_2026-04-12.py` -
  exact server script recovered on 2026-04-30.
- `results_lambda_100_N120_P1229.csv` - 1229 primes at `lambda=100`.
- `results_lambda_200_N120_P1229.csv` - 1229 primes at `lambda=200`.
- `prolate_run_N200_P10000_2026-04-12.log` - server production log.

Note: the two production CSV filenames contain the legacy marker `N120`, but
their hashes match the live server outputs from the N=200 production script.
The filenames are retained to avoid breaking existing references.

## Key Result

Production run:

- `lambda=100`: 1229/1229 primes have `Delta_p < 0`; `sum_Delta=-81.1992511503228`,
  `sum_HS_off=592.079620701508`, gap estimate `-673.278871851831`.
- `lambda=200`: 1229/1229 primes have `Delta_p < 0`; `sum_Delta=-115.23100013711`,
  `sum_HS_off=1046.52467709523`, gap estimate `-1161.75567723234`.

This confirms the Shift Parity Lemma's cumulative behavior in the full Galerkin
space for these two lambda values: no tested prime prefers the odd sector.

## Hash Audit (2026-04-30)

- Server N=200 script: `572539927ca22593369a73a614529947e0e70d1b8fbdcc7c4df4b0a784dcc66c`
- N120 log: `f3fcb158676743d810d887750d199554fa6faf71e6f4d826a42bed2f951f49a3`
- N200 log: `c4cd83496a610de92f37df62ead137f300cafa4bd5efbb6ab597e107532b8be7`
- `results_lambda_100_N120_P1229.csv`: `f638c915c283750296d3fa30ab05be6121854096e018163b812f8d3e803f92c3`
- `results_lambda_200_N120_P1229.csv`: `453e66a73853aa81673d80c7f8134b5a4d1371dd42df6ed9bed3c429f9ac0844`

## Reproduce

Dependencies: numpy, scipy, mpmath, tqdm.

```bash
PYTHONIOENCODING=utf-8 python prolate_prime_verification_exact_kernel_server_N200_P10000_2026-04-12.py
```

Runtime on ellmos-services was roughly 1-2 hours for the production run.

## Credit

Script authored by Gemini (Google DeepMind, assisting the author's research).
Archived in this project as supporting evidence for the RH Landscape audit trail.
