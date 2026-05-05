"""
pytest fixtures: synthetic PDF files for tests that don't require the corpus.
"""

from pathlib import Path

import fitz
import pytest


def _build_pdf(path: Path, pages: list[str], toc: list[list]) -> None:
    """
    Create a PDF file with the given pages and table of contents.

    Parameters
    ----------
    path : Path
        Path where the file is saved.
    pages : list[str]
        Text of each page.
    toc : list[list]
        Table of contents in PyMuPDF format: ``[[level, title, page], ...]``.
    """

    doc = fitz.open()
    try:
        for content in pages:
            page = doc.new_page()
            page.insert_text((72, 72), content)

        doc.set_toc(toc)
        doc.save(str(path))
    finally:
        doc.close()


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    """
    Single PDF with a two-level table of contents and a nested subsection.

    Returns
    -------
    Path
        Path to the created PDF file.
    """

    pdf_path = tmp_path / "sample.pdf"

    _build_pdf(
        pdf_path,
        pages=[
            "Intro text with hello word.",
            "Getting Started page content.",
            "Reference chapter starts here.",
            "Reference details on page four.",
        ],
        toc=[
            [1, "Introduction", 1],
            [2, "Getting Started", 2],
            [1, "Reference", 3],
            [2, "Details", 4],
        ],
    )

    return pdf_path


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """
    Directory containing two PDFs, each with its own table of contents.

    Returns
    -------
    Path
        Path to the corpus directory.
    """

    corpus = tmp_path / "corpus"
    corpus.mkdir()

    _build_pdf(
        corpus / "alpha.pdf",
        pages=["Alpha page 1.", "Alpha page 2."],
        toc=[[1, "Alpha Chapter", 1], [2, "Alpha Section", 2]],
    )
    _build_pdf(
        corpus / "beta.pdf",
        pages=["Beta page 1."],
        toc=[[1, "Beta Chapter", 1]],
    )

    return corpus
