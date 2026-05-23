# Zitationscheck 2026-05-17 -- RH Even Dominance Proof

## Projekt

- Projekt: `PP__RH_The Landscape`
- Paper: `RH_Even_Dominance_Proof_v1`
- Dateien: `paper/RH_Even_Dominance_Proof_v1_en.tex`, `paper/RH_Even_Dominance_Proof_v1_ger.tex`, `paper/references.bib`

## Auswahlgrund

Das Paper hatte im zentralen Registry noch keinen aktuellen Zitationscheck. Ein Scan der aktiven LaTeX-Dateien fand in beiden Sprachfassungen einen realen Fehler: `GeigerZookeeper2026` wurde zitiert, war aber nicht in `references.bib` enthalten.

## Befund

- EN und DE zitierten jeweils 10 eindeutige Keys.
- `references.bib` enthielt vor der Korrektur nur 9 Einträge.
- Fehlender Key in beiden Sprachfassungen: `GeigerZookeeper2026`.
- Der Zookeeper-Record wurde gegen die Zenodo-API geprüft: live Record `20151122`, DOI `10.5281/zenodo.20151122`, Version `1.1`, Publikationsdatum `2026-05-13`, Concept-DOI `10.5281/zenodo.19673126`.

## Korrektur

In `paper/references.bib` wurde ein BibTeX-Eintrag `GeigerZookeeper2026` ergänzt. Als DOI wird die stabile Concept-DOI `10.5281/zenodo.19673126` verwendet; die aktuell verifizierte Live-Version ist im `note`-Feld dokumentiert.

## Verifikation

- EN: 10 Cite-Keys, 10 BibTeX-Einträge, 10 BBL-Items; keine fehlenden oder unzitierten Keys.
- DE: 10 Cite-Keys, 10 BibTeX-Einträge, 10 BBL-Items; keine fehlenden oder unzitierten Keys.
- EN/DE wurden mit `pdflatex -> bibtex -> pdflatex -> pdflatex` neu gebaut.
- Die Kombi-PDF wurde per `pypdf` aus EN+DE neu gemergt.
- Finaler Logscan: keine Citation-Warnings, keine undefined references und keine Rerun-Warnings.
- Restwarnungen außerhalb des Zitationsfixes: EN 5 Overfull-HBox-Warnungen; DE 11 Overfull-HBox-Warnungen und 2 Warnungen zu doppelt definiertem Label `sec:tail`.
- `pdftotext -enc UTF-8` bestätigt echte deutsche Umlaute, u. a. `Unabhängiger`, `Brücke`, `Schlüsselwörter`, `Übergaben`, `für`.

## Artefakte

- EN-PDF: 22 Seiten, MD5 `B5E630A0F619D98782E5E76B575105C1`
- DE-PDF: 21 Seiten, MD5 `AAFBBB7A28ABCDACBD4EAB30C84FC539`
- Kombi-PDF: 43 Seiten, MD5 `03DFBADE5DFF90050B7E6A751C110E7E`

## Ergebnis

Der Zitationsfehler ist korrigiert. EN und DE sind zitationstechnisch synchron und vollständig aufgelöst. Kein Zenodo-Upload und kein GitHub-Sync wurden in diesem Lauf durchgeführt.
