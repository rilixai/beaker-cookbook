"""DOCX citations must survive extraction into the rubric judge's input."""

from pathlib import Path
from typing import Any

import pytest
from docx import Document
from docx.opc.constants import CONTENT_TYPE as CT
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.opc.packuri import PackURI
from docx.opc.part import Part
from docx.oxml import OxmlElement
from docx.oxml.ns import nsdecls, qn

from harvey_lab.agent.workspace import TaskWorkspace, extract_text
from harvey_lab.evaluation.scoring import score_rubric


def add_reference(paragraph: Any, kind: str, note_id: int) -> None:
    reference = OxmlElement(f"w:{kind}Reference")
    reference.set(qn("w:id"), str(note_id))
    paragraph.add_run()._r.append(reference)


def add_notes(document: Any, kind: str, contents: str) -> None:
    relationship, content_type = {
        "footnote": (RT.FOOTNOTES, CT.WML_FOOTNOTES),
        "endnote": (RT.ENDNOTES, CT.WML_ENDNOTES),
    }[kind]
    # A non-default package path verifies that extraction follows relationships.
    part = Part(
        PackURI(f"/word/notes/{kind}-citations.xml"),
        content_type,
        f"<w:{kind}s {nsdecls('w')}>{contents}</w:{kind}s>".encode(),
        document.part.package,
    )
    document.part.relate_to(part, relationship)


def test_footnote_and_endnote_citations_reach_the_judge_with_body_references(tmp_path: Path) -> None:
    workspace = TaskWorkspace(tmp_path)
    document = Document()
    add_reference(document.add_paragraph("Patent analysis"), "footnote", 1)
    add_reference(document.add_paragraph("Same patent authority"), "footnote", 1)
    cell = document.add_table(rows=1, cols=1).cell(0, 0)
    cell.paragraphs[0].text = "Tax opinion"
    add_reference(cell.paragraphs[0], "endnote", 1)
    add_notes(
        document,
        "footnote",
        """
        <w:footnote w:id="-1" w:type="separator"><w:p><w:r><w:t>Layout separator</w:t></w:r></w:p></w:footnote>
        <w:footnote w:id="0" w:type="continuationSeparator"><w:p><w:r><w:t>Continuation separator</w:t></w:r></w:p></w:footnote>
        <w:footnote w:id="1">
          <w:p><w:r><w:footnoteRef/></w:r><w:r><w:rPr><w:i/></w:rPr><w:t>Column 7, lines 22-38.</w:t></w:r></w:p>
          <w:p><w:hyperlink w:anchor="patent"><w:r><w:t>Patent specification.</w:t></w:r></w:hyperlink></w:p>
        </w:footnote>
        <w:footnote w:id="99"><w:p><w:r><w:t>Unreferenced citation</w:t></w:r></w:p></w:footnote>
        """,
    )
    add_notes(
        document,
        "endnote",
        """
        <w:endnote w:id="1" w:type="normal">
          <w:p><w:r><w:endnoteRef/></w:r><w:r><w:t>IRC Section 355(b)(2)(C).</w:t></w:r></w:p>
          <w:tbl><w:tblPr/><w:tblGrid><w:gridCol w:w="2400"/></w:tblGrid>
            <w:tr><w:tc><w:p><w:r><w:rPr><w:b/></w:rPr><w:t>IRC Section 355(b)(2)(D).</w:t></w:r></w:p></w:tc></w:tr>
          </w:tbl>
        </w:endnote>
        """,
    )
    document.save(workspace.deliverable_path("opinion.docx"))
    deliverables = workspace.collect_deliverables(["opinion.docx"])
    text = deliverables["opinion.docx"]
    assert "Same patent authority[footnote:1]" in text
    assert text.count("Column 7, lines 22-38.") == 1
    assert "separator" not in text
    assert "Unreferenced citation" not in text

    expected = {
        "patent": [
            "Patent analysis[footnote:1]",
            "[footnote:1]\n[italic]Column 7, lines 22-38.[/italic]\nPatent specification.",
        ],
        "tax": [
            "Tax opinion[endnote:1]",
            "[endnote:1]\nIRC Section 355(b)(2)(C).\n[bold]IRC Section 355(b)(2)(D).[/bold]",
        ],
    }

    def judge(_task: str, criteria: list[dict], output: str) -> dict[str, bool]:
        return {c["id"]: all(fragment in output for fragment in expected[c["id"]]) for c in criteria}

    result = score_rubric(
        criteria=[
            {"id": key, "match_criteria": "Include the required citations.", "deliverables": ["opinion.docx"]}
            for key in expected
        ],
        deliverables=deliverables,
        task_description="Write the legal analysis with citations.",
        judge=judge,
    )
    assert result["passed"] == 2


@pytest.mark.parametrize("kind", ["footnote", "endnote"])
@pytest.mark.parametrize("with_part", [False, True])
def test_missing_note_keeps_the_body_and_marks_missing_evidence(tmp_path: Path, kind: str, with_part: bool) -> None:
    document = Document()
    add_reference(document.add_paragraph("Analysis"), kind, 9)
    if with_part:
        add_notes(document, kind, f'<w:{kind} w:id="1"><w:p><w:r><w:t>Unrelated note</w:t></w:r></w:p></w:{kind}>')
    path = tmp_path / "opinion.docx"
    document.save(path)

    text = extract_text(path)
    assert text == f"Analysis[{kind}:9]\n[{kind}:9]\nNote text unavailable."
