"""Metadata, documentation, and manifest parity test suite for rh-even-dominance."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_core_documentation_files_exist() -> None:
    required_docs = [
        "README.md",
        "README_de.md",
        "SECURITY.md",
        "CITATION.cff",
        "llms.txt",
        "CHANGELOG.md",
        ".gitignore",
        "pyproject.toml",
        "LICENSE",
        "NOTICE",
    ]
    for doc in required_docs:
        doc_path = ROOT / doc
        assert doc_path.is_file(), f"Required core document missing: {doc}"
        assert doc_path.stat().st_size > 0, f"Document is empty: {doc}"


def test_paper_manuscripts_and_pdfs_exist() -> None:
    manuscripts = [
        "RH_I_Foundations",
        "RH_II_Even_Dominance",
        "RH_III_SpectralPaths",
        "RH_IV_CrossRoute",
        "RH_V_Conclusio",
    ]
    paper_dir = ROOT / "paper"
    assert paper_dir.is_dir(), "paper directory missing"

    for base in manuscripts:
        tex_file = paper_dir / f"{base}.tex"
        pdf_file = paper_dir / f"{base}.pdf"
        assert tex_file.is_file(), f"TeX file missing: {tex_file}"
        assert pdf_file.is_file(), f"PDF file missing: {pdf_file}"
        assert tex_file.stat().st_size > 500, f"TeX file too small: {tex_file}"
        assert pdf_file.stat().st_size > 10_000, f"PDF file too small: {pdf_file}"


def test_citation_cff_integrity() -> None:
    cff_path = ROOT / "CITATION.cff"
    assert cff_path.is_file(), "CITATION.cff missing"
    content = cff_path.read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in content
    assert 'family-names: "Geiger"' in content
    assert 'given-names: "Lukas"' in content
    assert "https://orcid.org/0009-0005-7296-1534" in content
    assert 'doi: "10.5281/zenodo.20479302"' in content
    assert 'repository-code: "https://github.com/research-line/rh-even-dominance"' in content
    assert 'license: "CC-BY-4.0"' in content


def test_llms_txt_integrity() -> None:
    llms_path = ROOT / "llms.txt"
    assert llms_path.is_file(), "llms.txt missing"
    content = llms_path.read_text(encoding="utf-8")

    assert "research-line/rh-even-dominance" in content
    assert "https://doi.org/10.5281/zenodo.19035640" in content
    assert "https://doi.org/10.5281/zenodo.20479302" in content
    assert "SECURITY.md" in content
    assert "README_de.md" in content
    assert ".github/workflows/tests.yml" in content
    assert "Last-checked: 2026-08-23" in content


def test_security_policy_integrity_and_bilingual_parity() -> None:
    sec_file = ROOT / "SECURITY.md"
    assert sec_file.is_file(), "SECURITY.md missing"
    content = sec_file.read_text(encoding="utf-8")

    assert "## English" in content
    assert "## Deutsch" in content
    assert "security@ellmos.ai" in content
    assert "support@lukasgeiger.com" in content
    assert "Zero-Egress" in content or "Zero--Egress" in content
    assert "User-Mode" in content or "Benutzerbereich" in content


def test_pyproject_pep621_metadata() -> None:
    toml_file = ROOT / "pyproject.toml"
    assert toml_file.is_file(), "pyproject.toml missing"
    content = toml_file.read_text(encoding="utf-8")

    assert 'name = "rh-even-dominance"' in content
    assert 'version = "3.1.2"' in content
    assert 'requires-python = ">=3.10"' in content
    assert "Programming Language :: Python :: 3.13" in content
    assert "Operating System :: OS Independent" in content
    assert "Topic :: Scientific/Engineering :: Mathematics" in content
    assert "[tool.ruff]" in content
    assert "[tool.pytest.ini_options]" in content


def test_ci_workflow_integrity() -> None:
    ci_file = ROOT / ".github" / "workflows" / "tests.yml"
    assert ci_file.is_file(), "GitHub Actions CI workflow file missing (.github/workflows/tests.yml)"
    content = ci_file.read_text(encoding="utf-8")

    assert "actions/checkout@v4" in content
    assert "actions/setup-python@v5" in content
    assert "ubuntu-latest" in content
    assert "windows-latest" in content
    for ver in ["3.10", "3.11", "3.12", "3.13"]:
        assert ver in content, f"Python version {ver} missing from CI matrix"
    assert "ruff check ." in content
    assert "pytest" in content


def test_result_certificates_json_validity() -> None:
    cert_dir = ROOT / "results" / "certificates"
    assert cert_dir.is_dir(), "results/certificates directory missing"

    for json_file in cert_dir.glob("*.json"):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data is not None, f"JSON file {json_file} loaded as None"
            assert isinstance(data, (dict, list)), f"JSON file {json_file} root must be dict or list"


def test_core_certification_scripts_byte_compile() -> None:
    import py_compile

    expected_scripts = [
        "certifier_production.py",
        "certifier_extended.py",
        "certifier_gap_closure.py",
        "certifier_simplicity.py",
        "euler_maclaurin_certifier.py",
        "certifier_lipschitz_analysis.py",
        "resolvent_analysis.py",
        "resolvent_R0K_test.py",
        "partA_bounded_diff.py",
        "partA_proof_sketch.py",
        "step4_gap_growth.py",
        "shift_parity_cert_v2.py",
        "shift_parity_cert_v3_targeted.py",
        "hellmann_feynman_gap.py",
        "endpoint_degeneracy.py",
        "subleading_gap.py",
        "verify_H1_schranke.py",
        "verify_lambda_star.py",
        "weighted_compactness_test.py",
        "weighted_compactness_server.py",
    ]
    scripts_dir = ROOT / "scripts"
    assert scripts_dir.is_dir(), "scripts directory missing"
    for script_name in expected_scripts:
        script_file = scripts_dir / script_name
        assert script_file.is_file(), f"Core script missing: {script_name}"
        assert script_file.stat().st_size > 0, f"Script is empty: {script_name}"
        compiled = py_compile.compile(str(script_file), doraise=True)
        assert compiled is not None


def test_readme_de_exists_and_bilingual_parity() -> None:
    readme_en = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_de = (ROOT / "README_de.md").read_text(encoding="utf-8")

    assert "# Vom Landscape zum Atlas" in readme_de
    assert "Forschungsstatus:" in readme_de
    assert "Schnelle Navigation" in readme_de
    assert "Quick Navigation" in readme_en
    assert "Geschwister-Forschungs- & Werkzeug-Ökosystem" in readme_de
    assert "Sibling Research & Tools Ecosystem" in readme_en
    assert "README_de.md" in readme_en
    assert "README.md" in readme_de


def test_mermaid_diagrams_syntax() -> None:
    for filename in ["README.md", "README_de.md"]:
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert "```mermaid" in content, f"Missing mermaid diagram block in {filename}"
        assert "flowchart TD" in content, f"Missing flowchart TD diagram in {filename}"
        assert "sequenceDiagram" in content, f"Missing sequence diagram in {filename}"


def test_sibling_ecosystem_and_urls() -> None:
    for filename in ["README.md", "README_de.md"]:
        content = (ROOT / filename).read_text(encoding="utf-8")
        assert "https://github.com/research-line/fst-nash" in content
        assert "https://github.com/research-line/functional-stability-theory" in content
        assert "https://github.com/ellmos-ai/open-compute-mcp" in content
        assert "https://github.com/doc-bricks/CleanMarkdown" in content
        assert "https://github.com/open-bricks" in content


def test_version_parity() -> None:
    toml_content = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    changelog_content = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    readme_content = (ROOT / "README.md").read_text(encoding="utf-8")
    readme_de_content = (ROOT / "README_de.md").read_text(encoding="utf-8")
    llms_content = (ROOT / "llms.txt").read_text(encoding="utf-8")

    version_match = re.search(r'version\s*=\s*"([^"]+)"', toml_content)
    assert version_match is not None, "Version not found in pyproject.toml"
    current_ver = version_match.group(1)
    assert current_ver == "3.1.2", f"Expected version 3.1.2, got {current_ver}"

    assert f"## [{current_ver}]" in changelog_content
    assert f"Version-{current_ver}-blue.svg" in readme_content
    assert f"Version-{current_ver}-blue.svg" in readme_de_content
    assert "Last-checked: 2026-08-23" in llms_content


def test_utf8_encoding_no_mojibake() -> None:
    extensions = {".md", ".py", ".tex", ".toml", ".txt", ".json", ".yml", ".yaml", ".cff"}
    target_paths = [
        *ROOT.glob("*"),
        *(ROOT / "scripts").rglob("*"),
        *(ROOT / "tests").rglob("*"),
        *(ROOT / "results" / "certificates").glob("*.json"),
        *(ROOT / "results" / "gap_analysis").glob("*.json"),
        *(ROOT / ".github").rglob("*"),
    ]
    for p in target_paths:
        if p.is_file() and p.suffix in extensions:
            if "_exploration" in p.parts:
                continue
            try:
                text = p.read_text(encoding="utf-8")
                assert "\ufffd" not in text, f"Mojibake (replacement char) detected in {p}"
            except UnicodeDecodeError as e:
                raise AssertionError(f"File {p} is not valid UTF-8: {e}") from e
