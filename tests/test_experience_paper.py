"""`paper_sources`: a paper's whole text becomes the research report the
pipeline ingests, from a file or a URL; a paper with nothing to read is
refused rather than ingested empty."""

import io
from types import SimpleNamespace

import pytest
from pypdf import PdfWriter

import kapso.experience.ingest as ingest_module
from kapso.experience import IngestError
from kapso.experience.ingest import paper_location, paper_sources

SETTINGS = {"paper_fetch_timeout_seconds": 7}


def pdf_bytes(text):
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
        f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = b"%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def test_a_markdown_paper_reaches_the_pipeline_whole(tmp_path):
    paper = tmp_path / "paper.md"
    text = "# A method\n" + "a long section of the paper\n" * 5000 + "the last line of the paper\n"
    paper.write_text(text)
    [report] = paper_sources(str(paper), SETTINGS)
    assert type(report).__name__ == "ResearchReport"
    assert report.query == str(paper) and report.content == text


def test_a_pdf_paper_is_read_to_text(tmp_path):
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(pdf_bytes("Low rank adapters on a quantized base"))
    [report] = paper_sources(str(paper), SETTINGS)
    assert "Low rank adapters on a quantized base" in report.content


def test_an_arxiv_abstract_link_is_fetched_as_its_pdf_within_the_configured_timeout(monkeypatch):
    fetched = []

    def get(url, timeout):
        fetched.append((url, timeout))
        return SimpleNamespace(
            content=pdf_bytes("Four bit normal float"),
            headers={"Content-Type": "application/pdf"},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(ingest_module.requests, "get", get)
    [report] = paper_sources("https://arxiv.org/abs/2305.14314v2", SETTINGS)
    assert fetched == [("https://arxiv.org/pdf/2305.14314v2", 7)]
    assert report.query == "https://arxiv.org/abs/2305.14314v2" and "Four bit normal float" in report.content


def test_a_paper_that_cannot_be_read_is_refused(tmp_path):
    with pytest.raises(IngestError, match="neither a file nor a URL"):
        paper_sources(str(tmp_path / "missing.pdf"), SETTINGS)
    blank = io.BytesIO()
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.write(blank)
    scanned = tmp_path / "scanned.pdf"
    scanned.write_bytes(blank.getvalue())
    with pytest.raises(IngestError, match="no text to ingest"):
        paper_sources(str(scanned), SETTINGS)


def reply(content, content_type):
    return SimpleNamespace(content=content, headers={"Content-Type": content_type}, raise_for_status=lambda: None)


def test_a_pdf_is_decided_by_its_content_not_its_header(monkeypatch):
    monkeypatch.setattr(
        ingest_module.requests, "get",
        lambda url, timeout: reply(pdf_bytes("Paged optimizers"), "application/octet-stream"),
    )
    [report] = paper_sources("https://host.example/files/paper", SETTINGS)
    assert "Paged optimizers" in report.content


def test_www_arxiv_abstract_links_are_rewritten_to_the_pdf(monkeypatch):
    fetched = []

    def get(url, timeout):
        fetched.append(url)
        return reply(pdf_bytes("Double quantization"), "application/pdf")

    monkeypatch.setattr(ingest_module.requests, "get", get)
    paper_sources("https://www.arxiv.org/abs/2305.14314", SETTINGS)
    assert fetched == ["https://arxiv.org/pdf/2305.14314"]


def test_an_html_reply_is_refused_naming_the_url_and_content_type(monkeypatch):
    url = "https://openreview.net/forum?id=OUIFPHEgJU"
    monkeypatch.setattr(
        ingest_module.requests, "get", lambda url, timeout: reply(b"<html>forum</html>", "text/html; charset=utf-8"),
    )
    with pytest.raises(IngestError, match=r"openreview\.net/forum\?id=OUIFPHEgJU.*text/html"):
        paper_sources(url, SETTINGS)


def test_a_plain_text_reply_is_ingested(monkeypatch):
    monkeypatch.setattr(
        ingest_module.requests, "get", lambda url, timeout: reply(b"A paper as text", "text/plain; charset=utf-8"),
    )
    [report] = paper_sources("https://host.example/paper.txt", SETTINGS)
    assert report.content == "A paper as text"


def test_a_local_file_that_is_not_a_pdf_markdown_or_text_is_refused(tmp_path):
    document = tmp_path / "paper.docx"
    document.write_bytes(b"PK\x03\x04\xff\xfe binary")
    with pytest.raises(IngestError, match=r"\.docx.*\.pdf, \.txt, \.md"):
        paper_sources(str(document), SETTINGS)


def test_a_relative_paper_path_is_recorded_resolved(tmp_path, monkeypatch):
    (tmp_path / "paper.md").write_text("text")
    monkeypatch.chdir(tmp_path)
    assert paper_location("paper.md") == str((tmp_path / "paper.md").resolve())
    assert paper_location("https://arxiv.org/abs/1") == "https://arxiv.org/abs/1"
