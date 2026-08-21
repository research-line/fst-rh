# Security Policy / Sicherheitsrichtlinie

---

## English

### Security & Research Integrity Policy

The `research-line/rh-even-dominance` repository is an open-science mathematical research and formal certification atlas mapping routes toward the Riemann Hypothesis via Connes' spectral program. We prioritize mathematical reproducibility, code safety, and data isolation.

### Supported Versions

| Version | Supported | Notes |
|---|---|---|
| `3.1.x` | :white_check_mark: | Current active Landscape/Atlas research and reproducibility baseline |
| `3.0.x` | :white_check_mark: | Five-part Atlas architecture baseline |
| `< 3.0.0` | :x: | Historical development drafts |

### Architectural Security Guarantees & Invariants

1. **100% Offline & Zero-Egress Execution:**
   - All analytical certifiers, interval arithmetic routines (mpmath.iv), and numerical verification scripts under `scripts/` and `tests/` execute entirely local-first.
   - Zero network egress, zero background telemetry, and zero remote data transmission during certification runs.
2. **User-Mode Non-Elevation:**
   - Verification suites, pytest runners, numerical certifiers, and LaTeX document compilers operate purely in standard user space.
   - No administrative, root, or elevated privileges are requested or required.
3. **Deterministic & Rigorous Reproducibility:**
   - All numerical certificates in `results/certificates/` use interval arithmetic bounds, Cauchy tail estimates, and deterministic Galerkin matrices.
   - Certificate outputs are structured JSON records with immutable numerical bounds.
4. **Data Isolation & Research Boundaries:**
   - Private research scratch notes (`BEWEISNOTIZ*.md`), internal working drafts (`_claude-work/`, `_proof-notes/`), and raw literature (`_sources/`) remain isolated and protected by strict `.gitignore` rules.

### Reporting a Vulnerability or Verification Anomaly

If you discover a security vulnerability, execution defect, or mathematical inconsistency in the certification suite:

1. **Email Reporting:** Please send details to **[security@ellmos.ai](mailto:security@ellmos.ai)** with a copy to **[support@lukasgeiger.com](mailto:support@lukasgeiger.com)**.
2. **GitHub Security Advisories:** Alternatively, report privately via [GitHub Private Vulnerability Reporting](https://github.com/research-line/rh-even-dominance/security/advisories/new).
3. **Response Timeline:** We acknowledge reports within 48 hours and provide a coordinated resolution plan within 7 business days.

---

## Deutsch

### Sicherheits- & Forschungsintegritäts-Richtlinie

Das Repository `research-line/rh-even-dominance` ist ein Open-Science-Forschungs- und formales Zertifizierungs-Atlas zur Kartierung von Wegen zur Riemannschen Vermutung über Connes' Spektralprogramm. Wir legen höchsten Wert auf mathematische Reproduzierbarkeit, Codesicherheit und Datenisolation.

### Unterstützte Versionen

| Version | Unterstützt | Anmerkungen |
|---|---|---|
| `3.1.x` | :white_check_mark: | Aktuelle aktive Landscape/Atlas-Forschungs- und Reproduzierbarkeitsbasis |
| `3.0.x` | :white_check_mark: | Fünfteilige Atlas-Architektur |
| `< 3.0.0` | :x: | Historische Entwicklungsstände |

### Architektonische Sicherheitsgarantien & Invarianten

1. **100% Offline & Zero-Egress Ausführung:**
   - Alle analytischen Zertifizierer, Intervallarithmetik-Routinen (mpmath.iv) und numerischen Prüfskripte in `scripts/` und `tests/` arbeiten vollständig lokal (Local-First).
   - Keinerlei Netzwerkübertragung, keine Hintergrund-Telemetrie und keine externe Datenerfassung während der Zertifizierungsläufe.
2. **User-Mode & Keine Administrator-Rechte:**
   - Testsuiten, Pytest-Läufe, Zertifizierer und LaTeX-Kompilierungsschritte laufen strikt im Standard-Benutzerbereich.
   - Es werden keine administrativen oder erhöhten Systemrechte angefordert oder benötigt.
3. **Deterministische & Strenge Reproduzierbarkeit:**
   - Alle numerischen Zertifikate in `results/certificates/` basieren auf Intervallarithmetik-Schranken, Cauchy-Restgliedabschätzungen und deterministischen Galerkin-Matrizen.
   - Zertifikatsausgaben sind strukturierte JSON-Dateien mit unveränderlichen mathematischen Intervallen.
4. **Datenschutz & Forschungsgrenzen:**
   - Private Notizen (`BEWEISNOTIZ*.md`), interne Arbeitsstände (`_claude-work/`, `_proof-notes/`) und Literaturquellen (`_sources/`) sind über strikte `.gitignore`-Regeln vom Versionskontrollfluss ausgeschlossen.

### Melden von Sicherheitslücken oder Verifikationsfehlern

Sollten Sie eine Sicherheitslücke oder eine mathematische Unstimmigkeit in den Zertifizierungsskripten entdecken:

1. **E-Mail-Meldung:** Senden Sie Ihren Bericht bitte an **[security@ellmos.ai](mailto:security@ellmos.ai)** (Kopie an **[support@lukasgeiger.com](mailto:support@lukasgeiger.com)**).
2. **GitHub Security Advisories:** Nutzen Sie alternativ das private [GitHub Security Advisory Reporting](https://github.com/research-line/rh-even-dominance/security/advisories/new).
3. **Reaktionszeit:** Sie erhalten innerhalb von 48 Stunden eine Eingangsbestätigung sowie binnen 7 Werktagen eine Rückmeldung mit Lösungsplan.
