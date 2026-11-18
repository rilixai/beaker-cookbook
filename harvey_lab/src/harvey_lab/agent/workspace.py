"""Per-task on-disk staging area for the Harvey LAB agent.

The agent itself does all of its file work inside a Stirrup code-execution
environment (one ``code_exec`` tool, as LAB-AA does), so this module is *not*
an agent toolbelt — it is the host-side staging area either end of that
environment:

* ``documents/`` — the task's input documents, materialized locally and then
  uploaded into the execution environment at session start.
* ``output/`` — where deliverables are pulled back *out* of the environment
  after the agent finishes, so the rubric judge can read them.

Because the agent can now emit real binary deliverables (``.docx`` /
``.xlsx`` / ``.pptx`` built with python-docx / openpyxl / python-pptx inside
the environment), :meth:`TaskWorkspace.collect_deliverables` extracts text
from whatever it finds rather than assuming UTF-8 — LAB-AA grades "the text
extracted from the criterion's declared deliverable files".

The parsers (python-docx / openpyxl / pypdf / python-pptx) are imported
lazily so the module stays importable offline; hermetic tests stage plain
text fixtures and never touch the heavy parsers.
"""

from __future__ import annotations

import email
import json
import logging
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)


# Per-case factory: ``(record) -> TaskWorkspace``. Backed by a local
# ``harvey-labs`` checkout; tests inject a fixture-backed one.
TaskSource = Callable[[Any], "TaskWorkspace"]


__all__ = [
    "TaskSource",
    "TaskWorkspace",
    "extract_text",
    "task_source_from_dir",
]


class TaskWorkspace:
    """A materialized ``documents/`` + ``output/`` staging tree for one task."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._documents = self._root / "documents"
        self._output = self._root / "output"
        self._documents.mkdir(parents=True, exist_ok=True)
        self._output.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def documents_dir(self) -> Path:
        return self._documents

    @property
    def output_dir(self) -> Path:
        return self._output

    def close(self) -> None:
        """Remove the workspace tree."""
        shutil.rmtree(self._root, ignore_errors=True)

    # ─── path safety ──────────────────────────────────────────────────

    def _resolve_within(self, base: Path, rel_path: str) -> Path:
        candidate = (base / rel_path).resolve()
        if candidate != base and base not in candidate.parents:
            raise ValueError(f"Path {rel_path!r} escapes {base.name}/.")
        return candidate

    # ─── deliverable retrieval (output/) ──────────────────────────────

    def deliverable_path(self, name: str) -> Path:
        """Absolute path a deliverable named ``name`` is pulled back down to."""
        return self._resolve_within(self._output, name)

    def collect_deliverables(self, names: Sequence[str] | None = None) -> dict[str, str]:
        """Extract text from the deliverables pulled back into ``output/``.

        ``names`` restricts collection to the exact filenames the task asked
        for (LAB-AA requires exact-filename matches). Passing ``None`` walks the
        whole ``output/`` tree instead — useful for inspection, not for grading.
        Files that are absent are simply omitted; the caller decides what a
        missing deliverable means for the rubric.
        """
        candidates: list[tuple[str, Path]] = []
        if names is None:
            candidates = [
                (path.relative_to(self._output).as_posix(), path)
                for path in sorted(self._output.rglob("*"))
                if path.is_file()
            ]
        else:
            for name in names:
                try:
                    candidates.append((name, self._resolve_within(self._output, name)))
                except ValueError:
                    logger.warning("Deliverable name %r escapes output/; skipped.", name)
        out: dict[str, str] = {}
        for name, path in candidates:
            if path.is_file():
                out[name] = extract_text(path)
        return out


# ─── lazy document parsers ────────────────────────────────────────────


def extract_text(path: Path | str, *, max_chars: int | None = None) -> str:
    """Extract gradable text from a file, dispatching on its extension.

    Handles the deliverable formats LAB tasks ask for (.docx / .xlsx / .pptx /
    .pdf / .eml) and falls back to a lenient UTF-8 decode for .md and other
    plain text. A parser failure returns a short note rather than raising, so
    one unreadable deliverable never aborts a grading run — the judge simply
    sees that the file could not be read.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    try:
        if suffix == ".docx":
            text = _read_docx(path)
        elif suffix in (".xlsx", ".xlsm"):
            text = _read_xlsx(path)
        elif suffix == ".pptx":
            text = _read_pptx(path)
        elif suffix == ".pdf":
            text = _read_pdf(path)
        elif suffix == ".eml":
            text = _read_eml(path)
        else:
            text = path.read_bytes().decode("utf-8", errors="replace")
    except Exception as exc:  # noqa: BLE001 - a bad file must not abort grading
        logger.warning("Could not extract text from %s: %s", path, exc)
        text = f"Could not extract text from {path.name!r} ({type(exc).__name__}: {exc})."
    return text if max_chars is None else text[:max_chars]


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("Text extraction requires the optional `python-docx` dependency for .docx.") from exc
    try:
        doc = Document(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        return f"Could not parse {path.name!r} as .docx ({type(exc).__name__}: {exc})."
    note_refs: dict[tuple[str, str], None] = {}
    parts = _docx_container_text(doc, note_refs)
    seen_parts: set[str] = set()
    for index, section in enumerate(doc.sections, start=1):
        for name in (
            "header",
            "first_page_header",
            "even_page_header",
            "footer",
            "first_page_footer",
            "even_page_footer",
        ):
            if name.startswith("first_page_") and not section.different_first_page_header_footer:
                continue
            if name.startswith("even_page_") and not doc.settings.odd_and_even_pages_header_footer:
                continue
            # Resolve linked content without creating missing parts. A later
            # section can enable a first-page header defined by an earlier one.
            for previous in reversed(doc.sections[:index]):
                container = getattr(previous, name)
                if not container.is_linked_to_previous:
                    break
            else:
                continue
            part_name = str(container.part.partname)
            if part_name in seen_parts:
                continue
            seen_parts.add(part_name)
            text = _docx_container_text(container, note_refs)
            if text:
                parts.extend([f"# Section {index} {name.replace('_', ' ')}", *text])
    parts.extend(_docx_notes_text(doc, note_refs))
    return "\n".join(parts)


def _docx_container_text(container: Any, note_refs: dict[tuple[str, str], None]) -> list[str]:
    return _docx_blocks_text(container.paragraphs, container.tables, note_refs)


def _docx_blocks_text(
    paragraphs: Sequence[Any], tables: Sequence[Any], note_refs: dict[tuple[str, str], None] | None = None
) -> list[str]:
    parts = []
    # Keep comment ranges in document order, including ranges crossing a table.
    blocks = sorted([*paragraphs, *tables], key=lambda block: block._element.getparent().index(block._element))
    for block in blocks:
        if hasattr(block, "rows"):
            for row in block.rows:
                cells = ["\n".join(_docx_blocks_text(c.paragraphs, c.tables, note_refs)).strip() for c in row.cells]
                if any(cells):
                    parts.append("\t".join(cells))
        elif (text := _docx_paragraph_text(block, note_refs)).strip():
            parts.append(text)
    return parts


def _docx_notes_text(doc: Any, note_refs: dict[tuple[str, str], None]) -> list[str]:
    """Resolve referenced notes through package relationships, retaining their body markers."""
    from docx.opc.constants import RELATIONSHIP_TYPE as RT
    from docx.oxml import parse_xml
    from docx.oxml.ns import qn
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    if not note_refs:
        return []
    notes: dict[tuple[str, str], str] = {}
    for kind, relationship in (("footnote", RT.FOOTNOTES), ("endnote", RT.ENDNOTES), ("comment", RT.COMMENTS)):
        for rel in doc.part.rels.values():
            if rel.reltype != relationship or rel.is_external:
                continue
            root = parse_xml(rel.target_part.blob)
            for note in root.findall(qn(f"w:{kind}")):
                note_id = note.get(qn("w:id"))
                key = (kind, note_id)
                if key not in note_refs or note.get(qn("w:type"), "normal") != "normal":
                    continue
                # Use the document's styles for note paragraphs and table cells.
                text = _docx_blocks_text(
                    [Paragraph(p, doc) for p in note.findall(qn("w:p"))],
                    [Table(table, doc) for table in note.findall(qn("w:tbl"))],
                )
                notes[key] = "\n".join(text)
    return [
        f"[{kind}:{note_id}]\n{notes.get((kind, note_id), 'Comment text unavailable.' if kind == 'comment' else 'Note text unavailable.')}"
        for kind, note_id in note_refs
    ]


def _docx_paragraph_text(paragraph: Any, note_refs: dict[tuple[str, str], None] | None = None) -> str:
    """Keep rubric-visible formatting and revisions in the judge's text input.

    Paragraph.text drops run styling and text inside tracked insertions/deletions.
    Walk inline XML so hyperlinks and revisions retain both text and markers.
    These markers describe document formatting; they are not grading verdicts.
    """
    from docx.oxml.ns import qn
    from docx.text.run import Run

    def fonts(run: Any) -> list[Any]:
        result = [run.font]
        for style in (run.style, paragraph.style):
            while style is not None:
                result.append(style.font)
                style = style.base_style
        return result

    def font_flag(run: Any, name: str) -> bool:
        for font in fonts(run):
            value = getattr(font, name)
            if value is not None:
                return bool(value)
        return False

    def font_color(run: Any) -> str | None:
        for font in fonts(run):
            properties = font._element.rPr
            color = properties.find(qn("w:color")) if properties is not None else None
            if color is not None:
                return " ".join(
                    f"{key}={color.get(qn('w:' + key))}"
                    for key in ("val", "themeColor", "themeTint", "themeShade")
                    if color.get(qn("w:" + key)) is not None
                )
        return None

    def inline(element: Any) -> str:
        if element.tag in {qn("w:commentRangeStart"), qn("w:commentRangeEnd")}:
            comment_id = element.get(qn("w:id"))
            if comment_id is not None:
                if note_refs is not None:
                    note_refs.setdefault(("comment", comment_id), None)
                boundary = "start" if element.tag == qn("w:commentRangeStart") else "end"
                return f"[comment:{comment_id}:{boundary}]"
        if element.tag == qn("w:r"):
            run = Run(element, paragraph)
            parts = []
            for child in element:
                if child.tag in {qn("w:t"), qn("w:delText")}:
                    parts.append(child.text or "")
                elif child.tag == qn("w:tab"):
                    parts.append("\t")
                elif child.tag in {qn("w:br"), qn("w:cr")}:
                    parts.append("\n")
                elif child.tag in {qn("w:footnoteReference"), qn("w:endnoteReference"), qn("w:commentReference")}:
                    kind = child.tag.rsplit("}", 1)[-1].removesuffix("Reference")
                    note_id = child.get(qn("w:id"))
                    if note_id is not None:
                        parts.append(f"[{kind}:{note_id}]")
                        if note_refs is not None:
                            note_refs.setdefault((kind, note_id), None)
            text = "".join(parts)
            if text and font_flag(run, "bold"):
                text = f"[bold]{text}[/bold]"
            if text and font_flag(run, "italic"):
                text = f"[italic]{text}[/italic]"
            if text and (font_flag(run, "strike") or font_flag(run, "double_strike")):
                text = f"[strikethrough]{text}[/strikethrough]"
            if text and font_flag(run, "underline"):
                text = f"[underline]{text}[/underline]"
            color = font_color(run)
            if text and color:
                text = f"[color {color}]{text}[/color]"
            return text
        text = "".join(inline(child) for child in element)
        if text and element.tag in {qn("w:ins"), qn("w:moveTo"), qn("w:del"), qn("w:moveFrom")}:
            kind = "inserted" if element.tag in {qn("w:ins"), qn("w:moveTo")} else "deleted"
            metadata = "".join(
                f" {key}={json.dumps(value, ensure_ascii=False)}"
                for key in ("author", "date", "id")
                if (value := element.get(qn("w:" + key))) is not None
            )
            return f"[{kind}{metadata}]{text}[/{kind}]"
        return text

    return inline(paragraph._p)


def _xlsx_visual_style(value: Any) -> str:
    """Keep colors and font indicators, including on otherwise empty cells."""

    def color_text(color: Any) -> str:
        if color is None:
            return ""
        return f"{color.type}:{color.value}" + (f" tint:{color.tint}" if color.tint else "")

    if hasattr(value, "has_style") and not value.has_style:
        return ""
    result = []
    fill = value.fill
    if fill is not None:
        if getattr(fill, "patternType", None):
            result.append(
                f"fill:{fill.patternType} foreground:{color_text(fill.fgColor)} background:{color_text(fill.bgColor)}"
            )
        elif getattr(fill, "stop", None):
            result.append("gradient:" + ",".join(f"{stop.position}={color_text(stop.color)}" for stop in fill.stop))
    font = value.font
    if getattr(value, "_style", None) is not None and value._style.fontId == 0:
        font = None
    if font is not None:
        if font.color is not None:
            result.append(f"font-color:{color_text(font.color)}")
        for name in ("bold", "italic", "strike", "underline"):
            flag = getattr(font, name)
            if flag:
                result.append(f"{name}:{flag}")
    return "; ".join(result)


def _read_xlsx(path: Path) -> str:
    from xml.etree.ElementTree import tostring

    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("Text extraction requires the optional `openpyxl` dependency for .xlsx.") from exc
    workbook = load_workbook(filename=str(path), read_only=False, data_only=False)
    try:
        cached = load_workbook(filename=str(path), read_only=True, data_only=True)
        try:
            names = list(workbook.sheetnames)
            chunks: list[str] = ["# Sheets: " + " | ".join(names)]
            for name in names:
                chunks.append(f"# Sheet: {name}")
                sheet = workbook[name]
                max_row = min(sheet.max_row or 200, 200)
                max_col = min(sheet.max_column or 40, 40)
                rows = sheet.iter_rows(max_row=max_row, max_col=max_col)
                values = cached[name].iter_rows(max_row=max_row, max_col=max_col, values_only=True)
                for row, cached_row in zip(rows, values, strict=True):
                    cells = []
                    for cell, value in zip(row, cached_row, strict=True):
                        if cell.data_type == "f":
                            formula = cell.value
                            # Array formulas store their expression on .text.
                            formula = getattr(formula, "text", formula)
                            result = (
                                "cached value unavailable; formula not evaluated"
                                if value is None
                                else f"cached value: {value}"
                            )
                            cells.append(f"{cell.coordinate}: {formula} [{result}]")
                        else:
                            cells.append("" if value is None else str(value))
                        style = _xlsx_visual_style(cell)
                        if style:
                            prefix = "" if cell.data_type == "f" else f"{cell.coordinate}: "
                            cells[-1] = f"{prefix}{cells[-1]} [{style}]"
                    chunks.append("\t".join(cells).rstrip("\t"))
                for condition in sheet.conditional_formatting:
                    for rule in sheet.conditional_formatting[condition]:
                        # Preserve the rule, not a claim about its rendered result. Excel is not run here.
                        styles = _xlsx_visual_style(rule.dxf) if rule.dxf is not None else ""
                        chunks.append(
                            f"Conditional formatting {condition.sqref} (not evaluated): "
                            f"{tostring(rule.to_tree(), encoding='unicode')} [{styles}]"
                        )
                if sheet.max_row and sheet.max_row > 200:
                    chunks.append("... (truncated rows)")
            return "\n".join(chunks).rstrip("\n")
        finally:
            cached.close()
    finally:
        workbook.close()


def _read_pdf(path: Path) -> str:
    head = path.read_bytes()[:5]
    if head[:4] == b"PK\x03\x04":
        return f"{path.name!r} is a ZIP/Office container mislabeled .pdf; read it as .docx/.xlsx instead."
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("Text extraction requires the optional `pypdf` dependency for .pdf.") from exc
    try:
        reader = PdfReader(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        return f"Could not parse {path.name!r} as a PDF ({type(exc).__name__}: {exc})."
    parts: list[str] = []
    for page_index, page in enumerate(reader.pages):
        if page_index >= 50:
            parts.append("... (truncated pages)")
            break
        try:
            parts.append(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - defensive per-page
            parts.append("")
    return "\n".join(parts)


def _read_pptx(path: Path) -> str:
    try:
        from pptx import Presentation
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError("Text extraction requires the optional `python-pptx` dependency for .pptx.") from exc
    presentation = Presentation(str(path))
    parts: list[str] = []
    for index, slide in enumerate(presentation.slides, start=1):
        parts.append(f"# Slide {index}")
        for shape in slide.shapes:
            text = getattr(shape, "text", "")
            if isinstance(text, str) and text.strip():
                parts.append(text)
            table = getattr(shape, "table", None) if getattr(shape, "has_table", False) else None
            if table is not None:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells]
                    if any(cells):
                        parts.append("\t".join(cells))
    return "\n".join(parts)


def _read_eml(path: Path) -> str:
    msg = email.message_from_bytes(path.read_bytes())
    headers = [f"{h}: {msg.get(h, '')}" for h in ("From", "To", "Cc", "Date", "Subject") if msg.get(h)]
    body_parts: list[str] = []
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if isinstance(payload, bytes):
                    body_parts.append(payload.decode("utf-8", errors="replace"))
    else:
        payload = msg.get_payload(decode=True)
        if isinstance(payload, bytes):
            body_parts.append(payload.decode("utf-8", errors="replace"))
    return "\n".join(headers) + "\n\n" + "\n".join(body_parts)


# ─── task-source factories ────────────────────────────────────────────


def task_source_from_dir(tasks_root: str | Path) -> TaskSource:
    """Build a ``(record) -> TaskWorkspace`` factory backed by a local tree.

    ``tasks_root`` is the ``tasks/`` directory of a ``harveyai/harvey-labs``
    checkout; a task's ``documents/`` live at ``tasks_root/<task_id>/documents``
    (``task_id`` is the task directory's path relative to ``tasks/``, possibly
    nested under a sub-category). The staged ``documents/`` tree is what gets
    uploaded into the agent's execution environment. Fully offline (used by the
    CLI + tests, which point it at a fixture tree).
    """
    base = Path(tasks_root)

    def _factory(record: Any) -> TaskWorkspace:
        import tempfile

        task_id = str(getattr(record, "task_id", "") or "")
        src_docs = base / task_id / "documents"
        ws = TaskWorkspace(tempfile.mkdtemp(prefix="harvey_lab_"))
        # Clean up the temp tree if the copy fails partway (don't leak on error).
        try:
            if src_docs.is_dir():
                shutil.copytree(src_docs, ws.documents_dir, dirs_exist_ok=True)
        except BaseException:
            ws.close()
            raise
        return ws

    return _factory
