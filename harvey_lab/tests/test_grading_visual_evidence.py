"""Deliverable annotations and visual evidence must reach the rubric judge."""

from pathlib import Path

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.shared import RGBColor
from openpyxl import Workbook
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Font, PatternFill

from harvey_lab.agent.workspace import TaskWorkspace, extract_text
from harvey_lab.evaluation.scoring import score_rubric


def assert_judge_receives(workspace: TaskWorkspace, filename: str, fragments: list[str]) -> str:
    deliverables = workspace.collect_deliverables([filename])

    def judge(_task, criteria, output):
        return {c["id"]: all(fragment in output for fragment in fragments) for c in criteria}

    result = score_rubric(
        criteria=[
            {
                "id": "visual",
                "match_criteria": "Include the annotations and visible markup.",
                "deliverables": [filename],
            }
        ],
        deliverables=deliverables,
        task_description="Prepare an annotated legal document.",
        judge=judge,
    )
    assert result["passed"] == 1
    return deliverables[filename]


def test_word_comments_retain_their_anchors_and_reach_the_judge(tmp_path: Path):
    workspace = TaskWorkspace(tmp_path)
    doc = Document()
    first = doc.add_paragraph().add_run("Loan principal")
    doc.add_table(rows=1, cols=1).cell(0, 0).text = "Collateral in the commented range"
    last = doc.add_paragraph().add_run("Revised maturity")
    comment = doc.add_comment([first, last], text="Conform to the commitment letter.", author="Reviewer")
    comment.add_paragraph("See section 2.1 of the term sheet.")
    table_run = doc.add_table(rows=1, cols=1).cell(0, 0).paragraphs[0].add_run("Interest rate")
    doc.add_comment(table_run, text="Address the client concern.", author="Reviewer")
    # An unanchored comment must not become evidence for a submitted edit.
    doc.comments.add_comment("Unattached rationale", author="Reviewer")
    doc.save(workspace.deliverable_path("markup.docx"))
    text = assert_judge_receives(
        workspace,
        "markup.docx",
        [
            "[comment:0:start]Loan principal",
            "Revised maturity[comment:0:end]",
            "[comment:0]\nConform to the commitment letter.\nSee section 2.1 of the term sheet.",
            "[comment:1:start]Interest rate[comment:1:end]",
            "[comment:1]\nAddress the client concern.",
        ],
    )
    assert (
        text.index("[comment:0:start]")
        < text.index("Collateral in the commented range")
        < text.index("[comment:0:end]")
    )
    assert "Unattached rationale" not in text
    assert text.count("Conform to the commitment letter.") == 1


def test_word_underline_and_colors_include_inherited_styles_and_overrides(tmp_path: Path):
    workspace = TaskWorkspace(tmp_path)
    doc = Document()
    style = doc.styles.add_style("Additions", WD_STYLE_TYPE.CHARACTER)
    style.font.underline = True
    style.font.color.rgb = RGBColor.from_string("FF0000")
    paragraph = doc.add_paragraph()
    paragraph.add_run("New covenant", style=style)
    override = paragraph.add_run("Existing covenant", style=style)
    override.font.underline = False
    override.font.color.rgb = RGBColor.from_string("000000")
    doc.save(workspace.deliverable_path("redline.docx"))
    text = assert_judge_receives(
        workspace,
        "redline.docx",
        [
            "[color val=FF0000][underline]New covenant[/underline][/color]",
            "[color val=000000]Existing covenant[/color]",
        ],
    )
    assert "[underline]Existing covenant" not in text


def test_xlsx_cell_colors_and_rules_reach_the_judge_without_evaluating_formulas(tmp_path: Path):
    workspace = TaskWorkspace(tmp_path)
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Compliance"
    sheet["A1"] = "State"
    sheet["A2"] = "New York"
    red = PatternFill("solid", fgColor="FFFF0000")
    sheet["B2"].fill = red  # A blank, colored gap indicator is still evidence.
    sheet["C2"] = "Gap"
    sheet["C2"].font = Font(color="FFFF0000", bold=True)
    sheet["D2"] = "=1+1"
    sheet.conditional_formatting.add("D2:D10", CellIsRule(operator="lessThan", formula=["3"], fill=red))
    workbook.save(workspace.deliverable_path("matrix.xlsx"))
    text = assert_judge_receives(
        workspace,
        "matrix.xlsx",
        [
            "State",
            "B2:  [fill:solid foreground:rgb:FFFF0000",
            "C2: Gap [font-color:rgb:FFFF0000; bold:True]",
            "D2: =1+1 [cached value unavailable; formula not evaluated]",
            "Conditional formatting D2:D10 (not evaluated)",
            'operator="lessThan"',
            "<formula>3</formula>",
        ],
    )
    assert "cached value: 2" not in text


def test_plain_spreadsheet_text_is_unchanged(tmp_path: Path):
    workbook = Workbook()
    workbook.active.append(["State", "Policy"])
    path = tmp_path / "plain.xlsx"
    workbook.save(path)
    assert extract_text(path) == "# Sheets: Sheet\n# Sheet: Sheet\nState\tPolicy"


def test_missing_word_comment_is_explicit_without_dropping_document(tmp_path: Path):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    doc = Document()
    run = doc.add_paragraph("Keep the document body.").add_run()
    marker = OxmlElement("w:commentReference")
    marker.set(qn("w:id"), "7")
    run._r.append(marker)
    path = tmp_path / "missing-comment.docx"
    doc.save(path)
    text = extract_text(path)
    assert text == "Keep the document body.[comment:7]\n[comment:7]\nComment text unavailable."
