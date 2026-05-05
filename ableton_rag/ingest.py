"""
Loading and preparing PDF documents for indexing.

Uses PyMuPDF to extract the table of contents (TOC) and per-page text.
Each TOC section becomes a LlamaIndex ``Document`` with metadata
(chapter, section, pages), which is then attached to every chunk produced
during indexing.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from llama_index.core import Document

from ableton_rag.config import get_logger, settings

logger = get_logger(__name__)


@dataclass
class Section:
    """
    Document section obtained from the table of contents.

    Attributes
    ----------
    title : str
        Section title.
    level : int
        Nesting level (1 = chapter, 2 = section, 3 = subsection).
    page_start : int
        First page.
    page_end : int
        Last page.
    parent_titles : list[str]
        Titles of the parent sections, from root to current.
    """

    title: str
    level: int
    page_start: int
    page_end: int
    parent_titles: list[str] = field(default_factory=list)

    @property
    def chapter(self) -> str:
        """Top-level chapter title."""
        if self.level == 1:
            return self.title

        return self.parent_titles[0] if self.parent_titles else ""

    @property
    def section(self) -> str:
        """Second-level section title."""
        if self.level <= 1:
            return ""

        if self.level == 2:
            return self.title

        return self.parent_titles[1] if len(self.parent_titles) > 1 else ""


def extract_toc(doc: fitz.Document) -> list[Section]:
    """
    Extract the table of contents and compute the page range for each section.

    Parameters
    ----------
    doc : fitz.Document
        Open PyMuPDF document.

    Returns
    -------
    list[Section]
        List of ``Section`` objects with their page ranges.
    """

    raw_toc = doc.get_toc()

    if not raw_toc:
        return []

    sections: list[Section] = []
    parent_stack: list[str] = []

    for i, (level, title, page) in enumerate(raw_toc):
        if i + 1 < len(raw_toc):
            page_end = raw_toc[i + 1][2] - 1
        else:
            page_end = doc.page_count

        parent_stack = parent_stack[: level - 1]
        parent_titles = list(parent_stack)
        parent_stack.append(title)

        sections.append(
            Section(
                title=title,
                level=level,
                page_start=page - 1,
                page_end=page_end,
                parent_titles=parent_titles,
            )
        )

    return sections


def clean_text(text: str) -> str:
    """
    Clean raw text extracted by PyMuPDF.

    Repairs hyphenated words at line breaks, removes page numbers
    and empty list bullets, and normalizes whitespace and line breaks.

    Parameters
    ----------
    text : str
        Raw text extracted by PyMuPDF.

    Returns
    -------
    str
        Cleaned text.
    """

    # Word hyphenation across line breaks
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Page numbers — lines consisting solely of digits
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Empty list bullets
    text = re.sub(r"^[•\-]\s*$", "", text, flags=re.MULTILINE)

    # Collapse three or more consecutive newlines into two
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Collapse runs of spaces
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def section_to_document(
    doc: fitz.Document, section: Section, source: str = ""
) -> Document | None:
    """
    Convert a TOC section into a LlamaIndex ``Document`` with metadata.

    Extracts the text of all pages in the section, cleans it and creates a
    ``Document`` with chapter/section/page metadata. LlamaIndex will split
    the ``Document`` into chunks during indexing according to
    ``Settings.chunk_size``.

    Parameters
    ----------
    doc : fitz.Document
        Open PyMuPDF document.
    section : Section
        Section with ``page_start`` and ``page_end`` attributes.
    source : str, optional
        Name of the source PDF (without extension). Stored in metadata.

    Returns
    -------
    Document or None
        ``Document`` with text and metadata, or ``None`` if the section is empty.
    """

    pages_text: list[str] = []

    for page_num in range(section.page_start, section.page_end + 1):
        if 0 <= page_num < doc.page_count:
            pages_text.append(doc[page_num].get_text())

    raw = "\n".join(pages_text)
    text = clean_text(raw)

    if not text.strip():
        return None

    return Document(
        text=text,
        metadata={
            "source": source,
            "chapter": section.chapter,
            "section": section.section,
            "subsection": section.title if section.level >= 3 else "",
            "toc_title": section.title,
            "toc_level": section.level,
            "page_start": section.page_start + 1,
            "page_end": section.page_end + 1,
        },
        excluded_llm_metadata_keys=["toc_level"],
    )


def load_documents(pdf_path: str | None = None) -> list[Document]:
    """
    Load the PDF corpus and create LlamaIndex Documents from TOC sections.

    Main entry point of the ingestion pipeline. Each TOC section becomes a
    separate ``Document`` with hierarchical metadata. LlamaIndex splits them
    into chunks when ``VectorStoreIndex.from_documents()`` is called.

    Parameters
    ----------
    pdf_path : str or None, optional
        Path to a PDF file or to a directory containing PDF files.
        Defaults to ``settings.corpus_path``.

    Returns
    -------
    list[Document]
        List of ``Document`` objects ready to be passed to ``VectorStoreIndex``.
    """

    root = Path(pdf_path) if pdf_path else settings.corpus_path
    pdf_files = sorted(root.glob("*.pdf")) if root.is_dir() else [root]

    logger.info("Loading documents from %s (%d PDF files)...", root, len(pdf_files))

    documents: list[Document] = []

    for pdf_file in pdf_files:
        doc = fitz.open(str(pdf_file))
        try:
            source = pdf_file.stem
            before = len(documents)

            for section in extract_toc(doc):
                llama_doc = section_to_document(doc, section, source=source)

                if llama_doc is not None:
                    documents.append(llama_doc)
        finally:
            doc.close()

        logger.info("  %s → %d sections", pdf_file.name, len(documents) - before)

    logger.info("Total documents extracted: %d", len(documents))

    return documents
