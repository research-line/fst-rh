# Changelog

All notable changes to the `rh-even-dominance` (From Landscape to Atlas) repository will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.2] - 2026-08-23

### Added
- **Interactive Mermaid Architecture & Lifecycle Diagrams:** Integrated high-resolution Mermaid flowchart (`flowchart TD`) mapping the 5-part Atlas structure, analytical main route A, spectral paths B/Z, and companion extractions; added sequence diagram (`sequenceDiagram`) visualizing the 50-digit interval-arithmetic verification lifecycle.
- **Bilingual Documentation (`README_de.md`):** Authored full German documentation parity package with German badges, tables, and localized Mermaid diagrams.
- **Sibling Research & Tools Ecosystem Matrix:** Documented integration with sibling open-science repositories (`research-line/fst-nash`, `functional-stability-theory`, `ai-elite-swr`), AI MCP infrastructure (`ellmos-ai/open-compute-mcp`, `codecommander`, `filecommander`), document processors (`doc-bricks/CleanMarkdown`, `DokuZen`), and `open-bricks` umbrella.
- **Expanded Metadata & Parity Contract Suite:** Extended `tests/test_metadata.py` from 10 to 14 contract tests covering bilingual documentation parity, Mermaid syntax, ecosystem references, and version consistency.

### Changed
- **Status & Discovery Badges:** Added direct links to `README_de.md`, updated test count to 14/14, and synchronized PEP 621 URLs in `pyproject.toml`.
- **LLM Context & Discovery:** Updated `llms.txt` with timestamp 2026-08-23, version 3.1.2, and expanded contract test counts.

---

## [3.1.1] - 2026-08-21

### Added
- **GitHub Actions CI Quality Gate:** Automated Multi-OS (`ubuntu-latest`, `windows-latest`) and Multi-Python (`3.10`, `3.11`, `3.12`, `3.13`) CI test matrix in `.github/workflows/tests.yml` with pip caching and ruff linter verification.
- **PEP 621 Metadata Standard:** Modernized `pyproject.toml` with PEP 621 classifiers (Python 3.10-3.13, OS Independent, Mathematics), project URLs, and standardized `[tool.ruff]` / `[tool.pytest.ini_options]` configuration.
- **Bilingual Security Policy:** Added `SECURITY.md` with explicit Local-First & Zero-Egress invariants, user-mode execution, deterministic interval arithmetic reproducibility, and direct security contact channels (`security@ellmos.ai` / `support@lukasgeiger.com`).
- **Automated Metadata & Integrity Test Suite:** Implemented `tests/test_metadata.py` with 10 comprehensive parity and contract tests (core documents, paper PDFs, CFF metadata, llms.txt, security policy, CI matrix, JSON certificates, and script compilation).
- **Modernized Status Badges:** Updated `README.md` with CI build status, Python versions, platform matrix, and Zero-Egress security badges.

### Changed
- **Crawler & LLM Context:** Updated `llms.txt` timestamp to 2026-08-21 with direct links to `SECURITY.md` and CI workflows.

---

## [3.1.0] - 2026-05-31

### Changed
- Repo sync of the Atlas pentalogy; companion Even-Dominance paper updated to v1.9 (`10.5281/zenodo.20479145`).
- Part I rebuilt for the five-part framing; combined English PDF regenerated (103 pages).
- Internal strategy and QA files safely isolated via `.gitignore`.

---

## [3.0.0] - 2026-05-26

### Changed
- Restructured the three-part Trilogy ("From Landscape to Proof") into the five-part Atlas ("From Landscape to Atlas").
- Part III split into Spectral Routes (III), Cross-Route Synthesis (IV), and Conclusio (V); structural reorganization only, conditional-reduction status unchanged.

---

## [2.5.0] - 2026-05-24

### Changed
- Public repository synced to the Part-III excluded-paths catalog extension.
- Part III EN/DE and combined PDFs rebuilt; local agent-control files kept private via `.gitignore`.

---

## [2.4.0] - 2026-05-23

### Added
- Parallel programme added to Part III.
- Companion Even-Dominance DOI updated to `10.5281/zenodo.20291994`.

---

## [2.3.0] - 2026-05-15

### Changed
- Conditioning clarified from unconditional closure to conditional reduction.
- Public paper package synced to the variational-gap revision.

---

## [2.1.0] - 2026-04-30

### Added
- Robust Direct Frontier-Dominance route.
- Gemini N=200 server-script archival and repository hygiene audit.

---

## [1.4.0] - 2026-03-27

### Changed
- Reviewer-driven clarifications (Prop A6 interpolation, M1'' explicit threshold, Lemma B Step 3/4 separation, Lemma L3 superseded, Galerkin safety margins, Connes2026 reference key).

---

## [1.3.0] - 2026-03-17

### Fixed
- Bibliographic corrections (Connes title, Deninger journal, Keiper type).

---

## [1.2.0] - 2026-03-16

### Added
- Interval arithmetic certifications (Euler-Maclaurin, OP2 simplicity, Lipschitz), explicit PNT bounds, and new verification scripts.

---

## [1.1.0] - 2026-03-15

### Added
- Lemma B/C analytical bounds; status upgraded to proved conditional steps.

---

## [1.0.0] - 2026-03-15

### Added
- Initial public release (A6 closed in variational framework, 33 numerical certificates).
