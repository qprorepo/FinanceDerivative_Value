# Manuscript

This directory contains the LaTeX sources for the project's academic
write-up, kept in three layers:

| File                                        | Role                                                                                                   | Status |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------|--------|
| `main_original.tex`                           | The original theory-only manuscript (derivations, theorems, algorithm descriptions) before any executed-simulation figures were merged in. | Reference / frozen |
| `new_main.tex`                                | The manuscript with the Data Availability section and glossary infrastructure added, in the process of having the 12 executed-simulation figures (`Figure_Captions_and_Explanations.tex`) merged into their correct sections. | **Work in progress** — see below |
| `glossary.tex`                                | `glossaries`-package acronym/term definitions (`\newacronym`, `\newglossaryentry`) shared by all three documents. | Stable |
| `Figure_Captions_and_Explanations.tex`        | A **standalone, independently-compiling** supplement containing all 12 executed-simulation figures with full captions and "Scientific Working Explanation" boxes, each citing real numbers copied verbatim from the notebook's captured output. | Complete, compiles cleanly |

## Compiling

All three top-level documents require **XeLaTeX** (not `pdflatex` — the
preamble uses `fontspec`/`unicode-math` for Unicode math support) plus
`biber` and `makeglossaries` for the full manuscript:

```bash
# Standalone figure supplement (no bibliography/glossary dependencies):
cd manuscript
xelatex Figure_Captions_and_Explanations.tex
xelatex Figure_Captions_and_Explanations.tex   # 2nd pass for cross-references
xelatex Figure_Captions_and_Explanations.tex   # 3rd pass to settle the TOC

# Full manuscript (needs bibliography.bib + glossary build step):
xelatex new_main.tex
biber new_main
makeglossaries new_main
xelatex new_main.tex
xelatex new_main.tex
```

