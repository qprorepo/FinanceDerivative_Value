"""
Shared publication-grade Matplotlib styling used by every figure in
``notebooks/QCMC_Quantum_Finance_Analysis.ipynb``.

This module intentionally contains only the *shared* styling/utility layer
(colour palette, rcParams, a ``commit_figure`` helper that saves a figure to
PNG + PDF and appends it to a running multi-page PDF report). The
figure-construction code itself (12 multi-panel figures, each combining
several of the ``qcmc`` engine modules) lives in the notebook, since it is
inherently narrative/exploratory rather than reusable library code -- see
``notebooks/`` and ``manuscript/figures/`` for the rendered output.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

__all__ = ["PALETTE", "apply_style", "FigureReport"]

#: Journal-grade colour palette, matching the TikZ/pgfplots palette used in
#: the LaTeX manuscript (``manuscript/new_main.tex``).
PALETTE = {
    "fblue": "#0F419B",
    "fgreen": "#14782D",
    "fred": "#B42323",
    "fpurple": "#691991",
    "forange": "#CD5F0F",
    "fcyan": "#0A91A5",
    "fgold": "#B9960A",
    "charcoal": "#2D2D2D",
    "midgray": "#828282",
    "lblue": "#D2E4FF",
    "lgreen": "#D7FFDC",
    "lred": "#FFDADA",
}


def apply_style() -> None:
    """Applies the shared rcParams used by every figure in the notebook."""
    matplotlib.use("Agg")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Georgia"],
            "mathtext.fontset": "dejavuserif",
            "axes.edgecolor": PALETTE["charcoal"],
            "axes.labelcolor": PALETTE["charcoal"],
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.2,
            "legend.fontsize": 7.6,
            "legend.frameon": True,
            "legend.framealpha": 0.9,
            "legend.edgecolor": PALETTE["midgray"],
            "grid.color": PALETTE["midgray"],
            "grid.alpha": 0.25,
            "grid.linewidth": 0.4,
            "axes.grid": True,
            "text.color": PALETTE["charcoal"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


class FigureReport:
    """Accumulates figures into a single multi-page PDF report, while also
    saving each one individually as PNG + PDF.

    Example
    -------
    >>> report = FigureReport("figures", "QCMC_Results_Report.pdf")
    >>> fig, ax = plt.subplots()
    >>> ax.plot([0, 1], [0, 1])
    >>> report.commit(fig, "sanity_check", "A trivial diagonal line.")
    >>> report.close()
    """

    def __init__(self, output_dir: str, report_name: str = "QCMC_Results_Report.pdf"):
        import os

        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.pdf_pages = PdfPages(os.path.join(output_dir, report_name))
        self.manifest: list[str] = []

    def commit(self, fig, name: str, caption: str = ""):
        import os

        fig.tight_layout()
        self.pdf_pages.savefig(fig, bbox_inches="tight")
        fig.savefig(os.path.join(self.output_dir, f"{name}.png"), bbox_inches="tight")
        fig.savefig(os.path.join(self.output_dir, f"{name}.pdf"), bbox_inches="tight")
        self.manifest.append(f"Fig. {len(self.manifest) + 1:02d} -- {name}: {caption}")
        return fig

    def close(self):
        self.pdf_pages.close()
