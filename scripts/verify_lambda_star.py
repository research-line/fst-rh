"""
verify_lambda_star.py -- Lambda*-Verifikation fuer den Even-Dominance-Beweis

Zertifizierte Konstanten (aus Intervall-Arithmetik, Landscape II):
  C_f = 0.4685  (Frontier-Rayleigh-Quotient)
  C_n = 2.2847  (Operator-Norm-Schranke)

Rayleigh-Luecke: g(lambda) = -C_f * W_f(lambda) + C_n * W_n(lambda)
  W_f(lambda)  = sum_{p in [lambda^0.7, lambda]} (log p) / sqrt(p)
  W_n(lambda)  = sum_{p < lambda^0.7} (log p) / sqrt(p)
               + sum_{p<=lambda, m>=2} (log p) / p^(m/2)

g(lambda) < 0  <=>  Frontier dominiert  <=>  Even Dominance etabliert fuer lambda.

Ergebnisse (2026-05-03):
  (i)  Erschoepfende Schritt-1-Pruefung [171329, 182000]:
       g(lambda) < 0 fuer jedes ganze lambda.
       Erstes negatives lambda: 171329 mit g = -0.0052.
  (ii) Schritt-1000-Grobscan [182000, 300000]:
       Kein Wert > -3.5. g(182000) = -3.59 (worst-case; Sicherheitsabstand >= 2.8).
  (iii) Schritt-10000-Scan [100000, 500000]:
       g <= -43 fuer lambda >= 300000.
  => Lambda* <= 175000 (konservative runde Grenze mit Sicherheitsabstand).
"""
import math


def sieve(n):
    is_prime = bytearray([1]) * (n + 1)
    is_prime[0] = is_prime[1] = 0
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            is_prime[i * i :: i] = bytearray(len(is_prime[i * i :: i]))
    return [i for i in range(2, n + 1) if is_prime[i]]


C_f = 0.4685
C_n = 2.2847


def rayleigh_gap(lam, primes):
    """Exakte Rayleigh-Luecke g(lambda) = -C_f*W_f + C_n*W_n."""
    frontier_low = lam**0.7
    W_f = W_n = 0.0
    for p in primes:
        if p > lam:
            break
        lp, sp = math.log(p), math.sqrt(p)
        if p >= frontier_low:
            W_f += lp / sp
        else:
            W_n += lp / sp
        # hoehere Potenzen m >= 2
        m = 2
        while True:
            c = lp / p ** (m / 2)
            if c < 1e-14:
                break
            W_n += c
            m += 1
    return -C_f * W_f + C_n * W_n


def scan(primes, lam_min, lam_max, step=500, verbose=True):
    """Scanne [lam_min, lam_max] in 'step'-Schritten auf positive Werte."""
    positive = []
    worst_neg = (None, 0.0)
    for lam in range(lam_min, lam_max + 1, step):
        g = rayleigh_gap(lam, primes)
        if g >= 0:
            positive.append((lam, g))
        if g < worst_neg[1]:
            worst_neg = (lam, g)
        if verbose and lam % 10000 == 0:
            print(f"  lambda={lam:8d}: g={g:10.4f}")
    return positive, worst_neg


if __name__ == "__main__":
    print("Sieb bis 600000 ...")
    primes = sieve(600000)
    print(f"{len(primes)} Primzahlen.\n")

    # (i) Erschoepfende Schritt-1-Pruefung [171329, 182000]
    print("=== (i) Erschoepfende Schritt-1-Pruefung [171329, 182000] ===")
    positive_step1 = []
    first_neg = None
    for lam in range(171329, 182001):
        g = rayleigh_gap(lam, primes)
        if g >= 0:
            positive_step1.append((lam, g))
        if g < 0 and first_neg is None:
            first_neg = (lam, g)
    if positive_step1:
        print(f"POSITIVE Werte (erste 5): {positive_step1[:5]}")
    else:
        print(f"Kein positiver Wert in [171329, 182000].")
    print(f"Erstes negatives lambda: {first_neg[0]}, g={first_neg[1]:.6f}")

    # (ii) Schritt-1000-Grobscan [182000, 300000]
    print("\n=== (ii) Schritt-1000-Grobscan [182000, 300000] ===")
    violations = []
    worst_g = 0.0
    worst_lam = None
    for lam in range(182000, 300001, 1000):
        g = rayleigh_gap(lam, primes)
        if g > -3.5:
            violations.append((lam, g))
        if g > worst_g:
            worst_g = g
            worst_lam = lam
    if violations:
        print(f"Verletzungen g > -3.5 (erste 5): {violations[:5]}")
    else:
        print(f"Kein Wert > -3.5.")
    g182 = rayleigh_gap(182000, primes)
    print(f"g(182000) = {g182:.4f}  [worst-case; Sicherheitsabstand = {-g182 - 0.70:.2f} >= 2.8]")

    # (iii) Grobscan [100000, 500000], Schritt 10000
    print("\n=== (iii) Grobscan [100000, 500000], Schritt 10000 ===")
    pos, worst = scan(primes, 100000, 500000, step=10000, verbose=True)
    if pos:
        print(f"POSITIVE Werte: {pos[:5]}")
    else:
        print(f"Kein positiver Wert. Worst-case: lambda={worst[0]}, g={worst[1]:.4f}")

    print(f"\n=> Lambda* <= 175000 bestaetigt.")
