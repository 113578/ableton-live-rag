"""
Загрузка и подготовка PDF-документа для индексирования.

Использует PyMuPDF для извлечения оглавления (TOC) и текста по страницам.
Каждый раздел TOC становится объектом LlamaIndex Document с метаданными
(глава, раздел, страницы), которые затем попадают в каждый чанк при индексировании.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz
from llama_index.core import Document

from ableton_live_rag.config import settings


@dataclass
class Section:
    """
    Раздел документа, полученный из оглавления.

    Attributes
    ----------
    title : str
        Заголовок раздела.
    level : int
        Уровень вложенности (1 = глава, 2 = раздел, 3 = подраздел).
    page_start : int
        Начальная страница.
    page_end : int
        Конечная страница.
    parent_titles : list[str]
        Заголовки родительских разделов от корня к текущему.
    """

    title: str
    level: int
    page_start: int
    page_end: int
    parent_titles: list[str] = field(default_factory=list)

    @property
    def chapter(self) -> str:
        """Название главы верхнего уровня."""
        if self.level == 1:
            return self.title

        return self.parent_titles[0] if self.parent_titles else ""

    @property
    def section(self) -> str:
        """Название раздела второго уровня."""
        if self.level <= 1:
            return ""

        if self.level == 2:
            return self.title

        return self.parent_titles[1] if len(self.parent_titles) > 1 else ""


def extract_toc(doc: fitz.Document) -> list[Section]:
    """
    Извлечение оглавления и вычисление диапазона страниц для каждого раздела.

    Parameters
    ----------
    doc : fitz.Document
        Открытый документ PyMuPDF.

    Returns
    -------
    list[Section]
        Список объектов Section с диапазонами страниц.
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
    Очищение сырого текста из PyMuPDF.

    Исправляет переносы слов на границах строк, удаляет номера страниц
    и пустые маркеры списков, нормализует пробелы и переносы строк.

    Parameters
    ----------
    text : str
        Необработанный текст, извлечённый PyMuPDF.

    Returns
    -------
    str
        Очищенный текст.
    """

    # Переносы слов
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Номера страниц — строки, состоящие только из цифр
    text = re.sub(r"^\d+\s*$", "", text, flags=re.MULTILINE)

    # Пустые маркеры списков
    text = re.sub(r"^[•\-]\s*$", "", text, flags=re.MULTILINE)

    # Схлопывание трёх подряд идущих переноса строки до двух
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Схлопывание множественных пробелов
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def section_to_document(
    doc: fitz.Document, section: Section, source: str = ""
) -> Document | None:
    """
    Преобразование раздела TOC в объект LlamaIndex Document с метаданными.

    Извлекает текст всех страниц раздела, очищает его и создаёт ``Document``
    с метаданными главы/раздела/страниц. LlamaIndex автоматически разобьёт
    ``Document`` на чанки при индексировании согласно ``Settings.chunk_size``.

    Parameters
    ----------
    doc : fitz.Document
        Открытый документ PyMuPDF.
    section : Section
        Раздел с атрибутами ``page_start`` и ``page_end``.
    source : str, optional
        Имя исходного PDF (без расширения). Записывается в метаданные.

    Returns
    -------
    Document or None
        ``Document`` с текстом и метаданными, или ``None`` если раздел пустой.
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
    Загрузка PDF-корпуса и создание списка LlamaIndex Documents из разделов TOC.

    Основная точка входа для пайплайна загрузки. Каждый раздел TOC
    становится отдельным ``Document`` с иерархическими метаданными.
    LlamaIndex сам разобьёт их на чанки при вызове
    ``VectorStoreIndex.from_documents()``.

    Parameters
    ----------
    pdf_path : str or None, optional
        Путь к PDF-файлу или директории с PDF-файлами. По умолчанию
        берётся из ``settings.corpus_path``.

    Returns
    -------
    list[Document]
        Список ``Document`` объектов, готовых к передаче в ``VectorStoreIndex``.
    """

    root = Path(pdf_path) if pdf_path else settings.corpus_path
    pdf_files = sorted(root.glob("*.pdf")) if root.is_dir() else [root]

    documents: list[Document] = []

    for pdf_file in pdf_files:
        doc = fitz.open(str(pdf_file))
        source = pdf_file.stem

        for section in extract_toc(doc):
            llama_doc = section_to_document(doc, section, source=source)
            if llama_doc is not None:
                documents.append(llama_doc)

        doc.close()

    return documents
