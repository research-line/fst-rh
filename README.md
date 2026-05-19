# From Landscape to Proof: The Even-Dominance Route to the Riemann Hypothesis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19035640.svg)](https://doi.org/10.5281/zenodo.19035640)

A three-part Landscape and audit-trail series for the even-dominance route to
the Riemann Hypothesis via Connes' spectral program (arXiv:2602.04022). It
documents the proof path, failed routes, obstruction analysis, computational
certificates, and the transition from the broader research landscape to the
separately published proof-only extraction.

Proof-only extraction: https://doi.org/10.5281/zenodo.20011910

This public repository is the curated reproducibility package for the
Landscape series. It intentionally includes the paper sources, public scripts,
and certificate outputs, while internal proof notebooks (`BEWEISNOTIZ*.md`,
`_proof-notes/`, handoffs, local server-root captures, and credentials) remain
private until a full project closeout permits release.

**Submitted to:** Communications in Mathematics (cm:17829, 2026-03-27)

## Current Status (v2.2, 2026-05-14)

The public paper package now reflects the v2.3 status correction. The finite
range `100 <= lambda <= 1,300,000` remains strong numerical evidence and a
conditional finite bridge; theorem-level finite even dominance still depends
on validating the odd-sector tail lower bound for `lambda_1^-`. The asymptotic
step is reduced to the `Asymptotic Variational Gap Conjecture`. The Direct
Frontier-Dominance argument remains part of the package as the asymptotic
variational route, but it no longer by itself closes the eigenvalue inequality
`lambda_1^+ < lambda_1^-`.

## Paper Series (3 Parts, EN + DE)

| Paper | File | Pages | Content |
|-------|------|-------|---------|
| **Part I** | `RH_I_Foundations` | 15 | Foundations and Obstructions: thermodynamic landscape (R1-R9), dead ends (K1-K4), reorientation to Connes |
| **Part II** | `RH_II_Even_Dominance` | 52 | **Main paper.** Shift Parity Lemma, 33 finite-range diagnostics, the M1'' variational framework, Leading-Mode Cancellation (c=2+sqrt(2)), Higher-Mode Decay (Lemma B), Resolvent Truncation (Lemma C), PNT Transfer, Euler-Maclaurin Proposition, Direct Frontier-Dominance, and the v2.3 status revision |
| **Part III** | `RH_III_Conclusio` | 19 | Synthesis of the proof architecture, revised v2.2 status table, explored alternatives (BI-1..11), independent results, and the remaining asymptotic gap problem |

All papers are available in English and German (DE suffix).
Combined English version: `paper/RH_Complete_Series_EN.pdf` (86 pages).

## Proof Architecture

| Step | Statement | Status |
|------|-----------|--------|
| A1 | Connes' Theorem 6.1 | proven (external) |
| A2 | Hurwitz sufficiency | proven (external) |
| A3 | Even dominance at 33 values (lambda=100..1.3M) | numerical evidence / conditional finite bridge |
| A4 | Shift Parity Lemma | **proven** |
| A5 | Frontier-prime mechanism | proven |
| **A6** | **Cumulative step** | **variational, v2.2** |
| A7 | Even dominance along `lambda_n -> infinity` | conditional (v2.2) |
| A8 | RH | conditional reduction (v2.2) |

## Key Results

1. **Shift Parity Lemma**: Every prime individually favors even eigenfunctions.
   Proved analytically (det/trace argument, Cauchy interlacing).

2. **33 finite-range gap diagnostics**: lambda = 100 to 1,300,000, with
   interval-arithmetic even upper bounds and a currently conditional odd
   lower-bound component.

3. **Leading-Mode Cancellation Lemma**: Overlap differences cancel pairwise with
   exact constant c = 2 + sqrt(2).

4. **M1'' Variational Framework**: The resolvent-damped comparison yields the
   asymptotic variational sign with explicit threshold lambda_0 = 442,413
   (Dusart bound).

5. **Current v2.3 status of Proposition A6**:
   - Regime 1 (lambda in [100, 1.3M]): 33 CAP diagnostics give strong
     finite-range evidence; theorem-level even dominance on the certified
     values requires validation of the odd-sector tail lower bound.
   - Regime 2 (asymptotic): M1'' + PNT Transfer + Lemma B + Lemma C identify
     the correct variational sign.
   - Remaining gap: asymptotic even dominance is reduced to the
     `Asymptotic Variational Gap Conjecture` for `lambda_1^-`.

6. **v2.1 Direct Frontier-Dominance**: independent asymptotic variational route
   using a common frontier Rayleigh vector, PNT partial summation, Mertens
   bounds, and finite diagnostic coverage through the explicit asymptotic threshold.
   It removes the earlier interpolation/PNT-constant caveats, but still
   requires a separate odd-sector lower bound to close the eigenvalue
   inequality.

7. **OP2 Simplicity**: Intra-even spectral gap certified by interval arithmetic
   at all 33 values (gap >= 8.69 at lambda=100, growing to >= 731 at lambda=320k).

## Scripts

### Core (scripts/)

| Script | Purpose |
|--------|---------|
| `certifier_production.py` | Production certifier: lambda 200-10000 |
| `certifier_extended.py` | Extended certifier: lambda 10000-640000 |
| `certifier_gap_closure.py` | Gap-closure certifier: lambda 700K-1.3M |
| `certifier_simplicity.py` | OP2 simplicity certification (interval arithmetic) |
| `euler_maclaurin_certifier.py` | Euler-Maclaurin IA certification (60-digit, 48-pt GL) |
| `certifier_lipschitz_analysis.py` | Gap-continuity / Lipschitz analysis |
| `resolvent_analysis.py` | Dense-grid resolvent energy analysis |
| `resolvent_R0K_test.py` | Neumann series convergence test |
| `partA_bounded_diff.py` | Mode decomposition of E_sin - E_cos |
| `partA_proof_sketch.py` | Overlap convergence analysis |
| `step4_gap_growth.py` | Block-bound gap prediction |
| `shift_parity_cert_v2.py` | Interval certification of Shift Parity |
| `shift_parity_cert_v3_targeted.py` | Targeted shift parity certification |
| `hellmann_feynman_gap.py` | Hellmann-Feynman derivative analysis |
| `endpoint_degeneracy.py` | Endpoint degeneracy analysis |
| `subleading_gap.py` | Subleading spectral gap analysis |
| `verify_H1_schranke.py` | H1 bound verification |
| `verify_lambda_star.py` | Exhaustive check of the lambda* threshold logic |
| `weighted_compactness_test.py` | Weighted compactness test |
| `weighted_compactness_server.py` | Server version of compactness test |

### Results (`results/`)

| File | Content |
|------|---------|
| `results/certificates/certificates.json` | 23 rigorous certificates (lambda 100-9201) |
| `results/certificates/certificates_extended.json` | 29 certificates (lambda 10000-320000) |
| `results/certificates/certificates_gap_closure.json` | 3 gap-closure certificates (700K, 1.05M, 1.3M) |
| `results/certificates/euler_maclaurin_results.json` | Euler-Maclaurin interval-arithmetic certification |
| `results/certificates/largeN_results.json` | Large-N certificate output |
| `results/certificates/rigorous_results.json` | Earlier rigorous certificate bundle |
| `results/certificates/rigorous_v3_lam100.json` | v3 lambda=100 certificate |
| `results/certificates/rigorous_v3_results.json` | v3 rigorous certificate summary |
| `results/certificates/rigorous_v4_lam100.json` | v4 lambda=100 certificate |
| `results/certificates/rigorous_v4_lam200.json` | v4 lambda=200 certificate |
| `results/certificates/simplicity_certificates.json` | OP2 simplicity certificates |
| `results/gap_analysis/gap_monotone_results.json` | Gap monotonicity analysis |
| `results/gap_analysis/gap_monotone_v2_results.json` | v2 gap monotonicity analysis |
| `results/gap_analysis/hellmann_feynman_results.json` | Hellmann-Feynman derivative analysis |
| `results/gap_analysis/lipschitz_analysis.json` | Gap-continuity Lipschitz analysis |
| `results/gap_analysis/resolvent_analysis.json` | Dense-grid resolvent energy analysis |

Historical exploration code in `scripts/_exploration/` and raw runtime logs in
`results/**/*.log` stay local-only on purpose; the tracked files are the
curated scripts and reproducible outputs needed for the public package.

### Gemini Verification (`scripts/gemini_verification/`)

Independent full-Galerkin checks at `lambda=100` and `lambda=200` are archived
with scripts, CSV outputs, and recovered server logs. The production run uses
`N=200`, `P_max=10000`, and confirms `1229/1229` tested primes deepen the gap
for both lambda values.

## Server Computation

Certificates are computed on ellmos-services (Hetzner CCX13, 2 vCPU, 8 GB RAM).
The certifier uses interval arithmetic (mpmath.iv, 50-digit precision) for the
even block and float64 with Cauchy tail bounds for the odd block.

## Version History

- **2.2** (2026-05-14): Status correction from unconditional closure to conditional reduction; public EN paper package synced to the variational-gap revision
- **2.1** (2026-04-30): Robust Direct Frontier-Dominance route, Gemini N=200 server-script archival, repo hygiene audit
- **1.4** (2026-03-27): Reviewer-driven clarifications (Prop A6 interpolation, M1'' explicit threshold, Lemma B Step 3/4 separation, Lemma L3 superseded, Galerkin safety margins, Connes2026 reference key)
- **1.3** (2026-03-17): Bibliographic corrections (Connes title, Deninger journal, Keiper type)
- **1.2** (2026-03-16): IA certifications (Euler-Maclaurin, OP2 simplicity, Lipschitz), explicit PNT bounds, new scripts
- **1.1** (2026-03-15): Lemma B/C analytical bounds, status upgrade to "proved"
- **1.0** (2026-03-15): Initial release (A6 closed, 33 certificates)

## Author

Lukas Geiger, Bernau, Germany
ORCID: [0009-0005-7296-1534](https://orcid.org/0009-0005-7296-1534)

---

## Haftung / Liability

Dieses Projekt ist eine **unentgeltliche Open-Source-Schenkung** im Sinne der §§ 516 ff. BGB. Die Haftung des Urhebers ist gemäß **§ 521 BGB** auf **Vorsatz und grobe Fahrlässigkeit** beschränkt. Ergänzend gelten die Haftungsausschlüsse aus GPL-3.0 / MIT / Apache-2.0 §§ 15–16 (je nach gewählter Lizenz).

Nutzung auf eigenes Risiko. Keine Wartungszusage, keine Verfügbarkeitsgarantie, keine Gewähr für Fehlerfreiheit oder Eignung für einen bestimmten Zweck.

This project is an unpaid open-source donation. Liability is limited to intent and gross negligence (§ 521 German Civil Code). Use at your own risk. No warranty, no maintenance guarantee, no fitness-for-purpose assumed.
