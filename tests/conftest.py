"""
Фикстуры pytest: синтетические PDF-файлы для тестов без зависимости от корпуса.
"""

from pathlib import Path

import fitz
import pytest


def _build_pdf(path: Path, pages: list[str], toc: list[list]) -> None:
    """
    Создание PDF-файла с заданными страницами и оглавлением.

    Parameters
    ----------
    path : Path
        Путь для сохранения файла.
    pages : list[str]
        Текст каждой страницы.
    toc : list[list]
        Оглавление в формате PyMuPDF: [[уровень, заголовок, страница], ...].
    """

    doc = fitz.open()

    for content in pages:
        page = doc.new_page()
        page.insert_text((72, 72), content)

    doc.set_toc(toc)
    doc.save(str(path))
    doc.close()


@pytest.fixture
def tiny_pdf(tmp_path: Path) -> Path:
    """
    Одиночный PDF с двухуровневым оглавлением и вложенным подразделом.

    Returns
    -------
    Path
        Путь к созданному PDF-файлу.
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
    Директория с двумя PDF, каждый со своим оглавлением.

    Returns
    -------
    Path
        Путь к директории корпуса.
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
