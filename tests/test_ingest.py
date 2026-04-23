"""
Тесты модуля ingest: очистка текста, свойства Section и загрузка документов.
"""

from pathlib import Path

from ableton_live_rag.ingest import Section, clean_text, load_documents


def test_clean_text_fixes_hyphenation_and_strips_page_numbers():
    raw = "This is a hy-\nphenated word.\n42\nNext line  has   extra    spaces.\n\n\n\nDone."
    cleaned = clean_text(raw)

    assert "hyphenated" in cleaned
    assert "hy-" not in cleaned
    assert "\n42\n" not in cleaned
    assert "42" not in cleaned.split()
    assert "extra spaces" in cleaned
    assert "\n\n\n" not in cleaned


def test_section_chapter_and_section_properties():
    chapter = Section(title="Ch", level=1, page_start=0, page_end=5)
    subsection = Section(
        title="Sub",
        level=3,
        page_start=0,
        page_end=5,
        parent_titles=["Ch", "Sec"],
    )

    assert chapter.chapter == "Ch"
    assert chapter.section == ""
    assert subsection.chapter == "Ch"
    assert subsection.section == "Sec"


def test_load_documents_single_pdf_metadata(tiny_pdf: Path):
    docs = load_documents(str(tiny_pdf))

    assert len(docs) == 4

    titles = [d.metadata["toc_title"] for d in docs]
    assert titles == ["Introduction", "Getting Started", "Reference", "Details"]

    intro = docs[0]
    assert intro.metadata["source"] == "sample"
    assert intro.metadata["chapter"] == "Introduction"
    assert intro.metadata["page_start"] == 1

    details = docs[3]
    assert details.metadata["chapter"] == "Reference"
    assert details.metadata["section"] == "Details"
    assert details.metadata["page_start"] == 4


def test_load_documents_from_directory_merges_all_pdfs(corpus_dir: Path):
    docs = load_documents(str(corpus_dir))

    sources = {d.metadata["source"] for d in docs}
    assert sources == {"alpha", "beta"}

    assert len(docs) == 3
