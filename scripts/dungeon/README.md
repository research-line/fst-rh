# Dungeon Scripts — Alternative Proof Route Experiments

Session: 2026-04-12 / 2026-04-13
Context: exploration of alternative proof paths for even dominance,
leading to the two non-existence theorems NE-A and NE-B (v2.0 of Paper II).

## Files

### `cos2x_root_identity.py`
Tests the hypothesis that the shift-parity difference matrix $D_{nm}(r)$ reduces
to a single Fourier mode via the identity $\cos^2 - \sin^2 = \cos(2x)$.

**Result:** The root identity is *softer* than hoped. The different basis choices
(index $n$ for cos, $n+1$ for sin) introduce boundary mixing; the difference is
not a single Fourier mode. Diagonal values $D_{nn}(1) = \pm 1$ are clean, but
off-diagonals oscillate.

Status: Informative, not used in final proof.

### `door_B_slepian_commutator.py`
Tests whether an approximation of Slepian's prolate spheroidal operator $P$
commutes with $D_N(r)$ for fixed $r$.

**Result:** Relative commutator norm $\|[P, D(r=1)]\|_F / (\|P\| \|D\|)$ decreases
with $N$: from 0.21 at $N=6$ to 0.0041 at $N=30$, approximately $N^{-2.4}$.
This is pointwise in $r$, not a universal commutator.

Status: Suggestive for prolate perturbation (Route C), but does not establish a
universal Sturm–Liouville structure.

### `door_B_universal_T.py`
Searches for a universal symmetric $T$ with $[T, D_N(r_i)] = 0$ for a dense
$r$-grid in $(0,2)$, via SVD on the stacked commutator constraint matrix.

**Result:** For $N \in \{3, 5, 7, 10, 15\}$, the kernel dimension is exactly 1,
and the sole solution is $T = c \cdot I$. The second smallest singular value
$\sigma_2$ stays bounded below $\sim\!0.70$ at $N=15$ and does not decrease
with $N$.

**This is the numerical evidence for Theorem NE-B (No Universal Commuting Operator).**

## How to reproduce

```bash
cd scripts/dungeon
PYTHONIOENCODING=utf-8 python door_B_universal_T.py
```

Dependencies: numpy, scipy (eigh, quad).

## Related external work

See `../gemini_verification/` for Gemini's full Galerkin-space (N=120) prime
verification at $\lambda=100$ and $\lambda=200$. Result: 303/303 primes
deepen the gap, no exceptions. Provides independent numerical support for
the Shift Parity Lemma in the full operator space.
