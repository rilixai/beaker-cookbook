"""Agent tools: ``fs_search``, ``fs_read``, ``python_exec`` and (opt-in) ``web_search``.

Capabilities follow the OfficeQA Pro report (arXiv:2603.08655) §4.1 and
Appendix D.1, Table 3 — "Mapping of fs_search and fs_read tool capabilities
to equivalent shell commands":

    fs_search  directory listing               ls
    fs_search  list files by pattern           ls treasury_bulletins/*2013*.txt
    fs_search  search directory recursively    grep -rn -i -B 2 -A 2 "national saving" treasury_bulletins/
    fs_search  search in specific files        grep -rn -i "Individual income taxes" treasury_bulletins/treasury_bulletin_2013*.txt
    fs_read    read first N lines              head -n 20 foo.txt
    fs_read    read a slice of lines           sed -n '101,150p' foo.txt
    fs_read    read entire file                cat foo.txt

§4.1: "These custom classes include functionality to ensure that the agent
cannot read outside of its current working directory, and doesn't read large
outputs (e.g., large files potentially returned by fs_read)." The prompt puts
the corpus at ``../officeqa_corpus/``, so "working directory" here means the
question's cwd plus the corpus tree; nothing else is reachable.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import select
import shutil
import subprocess
import textwrap
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from officeqa import config
from officeqa.data.corpus import Workspace


logger = logging.getLogger(__name__)

ToolFn = Callable[..., Awaitable[str]]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


# --- Path confinement -------------------------------------------------------


class PathOutsideWorkspace(PermissionError):
    pass


def confine(user_path: str, workspace: Workspace) -> Path:
    """Resolve ``user_path`` relative to the cwd; reject anything outside the allowed roots."""
    candidate = Path(user_path)
    if not candidate.is_absolute():
        candidate = workspace.cwd / candidate
    resolved = candidate.resolve()
    allowed = list(workspace.allowed_roots) + [workspace.corpus_link.resolve()]
    for root in allowed:
        if resolved == root or root in resolved.parents:
            return resolved
    raise PathOutsideWorkspace(
        f"{user_path!r} is outside the working directory. Reachable paths: the current directory (.) "
        f"and the corpus at ../{config.CORPUS_ROOT_NAME}/ ({', '.join(f'{n}/' for n in config.CORPUS_REPRESENTATIONS)})."
    )


def _display(path: Path, workspace: Workspace) -> str:
    """Agent-relative rendering of a resolved path (``../officeqa_corpus/pdfs/x.pdf`` or ``notes.txt``)."""
    cwd = workspace.cwd.resolve()
    if path == cwd or cwd in path.parents:
        return str(path.relative_to(cwd)) or "."
    corpus = workspace.corpus_link
    for name in config.CORPUS_REPRESENTATIONS:
        sub = corpus / name
        if sub.is_dir():
            real = sub.resolve()
            if path == real or real in path.parents:
                return f"../{config.CORPUS_ROOT_NAME}/{name}/{path.relative_to(real)}".rstrip("/")
    corpus_real = corpus.resolve()
    if path == corpus_real:
        return f"../{config.CORPUS_ROOT_NAME}"
    if corpus_real in path.parents:
        return f"../{config.CORPUS_ROOT_NAME}/{path.relative_to(corpus_real)}"
    return str(path)


# --- fs_search --------------------------------------------------------------


def _human(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}GB"


def _list_dir(directory: Path, pattern: str | None, workspace: Workspace, max_results: int) -> str:
    if directory.is_file():
        entries = [directory]
    elif pattern:
        entries = sorted(directory.glob(pattern))
    else:
        entries = sorted(directory.iterdir())
    total = len(entries)
    lines = []
    for p in entries[:max_results]:
        if p.is_dir():
            lines.append(f"{_display(p, workspace)}/")
        else:
            try:
                size = _human(p.stat().st_size)
            except OSError:
                size = "?"
            lines.append(f"{_display(p, workspace)}\t{size}")
    if total > max_results:
        lines.append(f"... {total - max_results} more entries not shown (raise max_results or narrow the pattern)")
    if not lines:
        return "(no matches)"
    return "\n".join(lines)


def _grep_python(
    files: Sequence[Path],
    query: str,
    *,
    ignore_case: bool,
    before: int,
    after: int,
    max_results: int,
    workspace: Workspace,
) -> str:
    flags = re.IGNORECASE if ignore_case else 0
    try:
        rx = re.compile(query, flags)
    except re.error:
        rx = re.compile(re.escape(query), flags)
    out: list[str] = []
    hits = 0
    for f in files:
        if hits >= max_results:
            break
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue
        if "\x00" in text[:4096]:
            continue
        lines = text.splitlines()
        shown = _display(f, workspace)
        for i, line in enumerate(lines):
            if rx.search(line):
                hits += 1
                lo, hi = max(0, i - before), min(len(lines), i + after + 1)
                for j in range(lo, hi):
                    sep = ":" if j == i else "-"
                    out.append(f"{shown}{sep}{j + 1}{sep}{lines[j]}")
                if before or after:
                    out.append("--")
                if hits >= max_results:
                    out.append(f"... stopped after {max_results} matches (narrow the query or set files/glob)")
                    break
    return "\n".join(out) if out else "(no matches)"


def _grep_rg(
    rg: str,
    files: Sequence[Path],
    query: str,
    *,
    ignore_case: bool,
    before: int,
    after: int,
    max_results: int,
    workspace: Workspace,
) -> str:
    cmd = [rg, "-n", "--no-heading", "--color=never", "-M", "2000", "--max-count", str(max_results)]
    if ignore_case:
        cmd.append("-i")
    if before:
        cmd += ["-B", str(before)]
    if after:
        cmd += ["-A", str(after)]
    cmd += ["-e", query, "--", *map(str, files)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(workspace.cwd))
    except subprocess.TimeoutExpired:
        return "(search timed out after 120s; narrow the query or set files/glob)"
    if proc.returncode == 2 and "regex parse error" in proc.stderr:
        cmd[cmd.index("-e")] = "-F"
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120, cwd=str(workspace.cwd))
    if proc.returncode not in (0, 1):
        return f"(search error) {proc.stderr.strip()[:2000]}"
    if not proc.stdout.strip():
        return "(no matches)"
    # rg prints absolute paths because we passed them; rewrite to agent-relative.
    lines = []
    for line in proc.stdout.splitlines()[: max_results * (1 + before + after) + 1]:
        m = re.match(r"^(/[^:\-]*?)([:-])(\d+)\2", line)
        if m:
            line = _display(Path(m.group(1)), workspace) + line[len(m.group(1)) :]
        lines.append(line)
    return "\n".join(lines)


def make_fs_search(workspace: Workspace) -> Tool:
    async def fs_search(
        path: str = ".",
        glob: str | None = None,
        query: str | None = None,
        ignore_case: bool = True,
        before: int = 0,
        after: int = 0,
        max_results: int = 100,
    ) -> str:
        try:
            target = confine(path, workspace)
        except PathOutsideWorkspace as e:
            return f"Error: {e}"
        if not target.exists():
            return f"Error: {path!r} does not exist."
        max_results = max(1, min(int(max_results), 2000))
        if query is None:
            return _list_dir(target, glob, workspace, max_results)
        if target.is_file():
            files: list[Path] = [target]
        else:
            files = sorted(p for p in (target.glob(glob) if glob else target.rglob("*")) if p.is_file())
        if not files:
            return "(no files to search)"
        rg = shutil.which("rg")
        if rg:
            return await asyncio.to_thread(
                _grep_rg,
                rg,
                files,
                query,
                ignore_case=ignore_case,
                before=int(before),
                after=int(after),
                max_results=max_results,
                workspace=workspace,
            )
        return await asyncio.to_thread(
            _grep_python,
            files,
            query,
            ignore_case=ignore_case,
            before=int(before),
            after=int(after),
            max_results=max_results,
            workspace=workspace,
        )

    return Tool(
        name="fs_search",
        description=(
            "Search the file system, confined to your working directory and the corpus at ../officeqa_corpus/. "
            "Without `query`: list `path` (like `ls`), optionally filtered by `glob` (like `ls dir/*2013*.txt`). "
            "With `query`: recursively grep `path` (a directory or a single file) for a regex or literal string "
            "(like `grep -rn -i -B before -A after query path`), optionally restricting to files matching `glob`. "
            "PDF files are binary and are skipped by search; extract their text with python_exec."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory or file, relative to the working directory. Default '.'",
                },
                "glob": {
                    "type": "string",
                    "description": "Filename pattern relative to `path`, e.g. '*2013*.txt' or '**/*.pdf'.",
                },
                "query": {
                    "type": "string",
                    "description": "Regex or literal to search for. Omit to list files instead.",
                },
                "ignore_case": {"type": "boolean", "description": "Case-insensitive search (default true)."},
                "before": {"type": "integer", "description": "Context lines before each match (grep -B)."},
                "after": {"type": "integer", "description": "Context lines after each match (grep -A)."},
                "max_results": {
                    "type": "integer",
                    "description": "Cap on matches / entries returned (default 100, max 2000).",
                },
            },
        },
        fn=fs_search,
    )


# --- fs_read ----------------------------------------------------------------


def make_fs_read(workspace: Workspace, *, output_limit: int = config.TOOL_OUTPUT_LIMIT) -> Tool:
    async def fs_read(path: str, start: int | None = None, end: int | None = None, head: int | None = None) -> str:
        try:
            target = confine(path, workspace)
        except PathOutsideWorkspace as e:
            return f"Error: {e}"
        if not target.is_file():
            return f"Error: {path!r} is not a file."
        raw = await asyncio.to_thread(target.read_bytes)
        if target.suffix.lower() == ".pdf" or b"\x00" in raw[:4096]:
            return (
                f"Error: {path!r} is a binary file ({_human(len(raw))}). fs_read only reads text. "
                "Use python_exec (e.g. pymupdf/fitz, pdfplumber, pypdf, or pytesseract for scanned pages) to extract text."
            )
        text = raw.decode("utf-8", errors="replace")
        lines = text.splitlines()
        n = len(lines)
        if head is not None:
            lo, hi = 0, max(0, int(head))
        elif start is not None or end is not None:
            lo = max(1, int(start or 1)) - 1
            hi = int(end) if end is not None else n
        else:
            lo, hi = 0, n
            if len(text) > output_limit:
                return (
                    f"Error: {path!r} is {n} lines / {len(text):,} characters, which exceeds the {output_limit:,}-character "
                    "read limit. Read a slice with start/end (like `sed -n 'start,endp'`) or head, or search it with fs_search."
                )
        hi = min(hi, n)
        chunk = lines[lo:hi]
        body = "\n".join(f"{i + 1}\t{line}" for i, line in enumerate(chunk, start=lo))
        if len(body) > output_limit:
            return (
                f"Error: lines {lo + 1}-{hi} of {path!r} are {len(body):,} characters, over the {output_limit:,}-character "
                "limit. Request a smaller slice."
            )
        header = f"{_display(target, workspace)} (lines {lo + 1}-{hi} of {n})"
        return f"{header}\n{body}" if body else f"{header}\n(empty)"

    return Tool(
        name="fs_read",
        description=(
            "Read a text file, confined to your working directory and the corpus at ../officeqa_corpus/. "
            "Use `head` for the first N lines (like `head -n N`), `start`/`end` (1-based, inclusive) for a slice "
            "(like `sed -n 'start,endp'`), or neither for the whole file (like `cat`). Whole-file reads over "
            f"{output_limit:,} characters are refused; read slices instead. Lines are prefixed with their number."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path relative to the working directory."},
                "start": {"type": "integer", "description": "First line to read (1-based)."},
                "end": {"type": "integer", "description": "Last line to read (inclusive)."},
                "head": {"type": "integer", "description": "Read only the first N lines."},
            },
            "required": ["path"],
        },
        fn=fs_read,
    )


# --- python_exec ------------------------------------------------------------

# Report §4.1: "The Python REPL tool provides the agent with a persistent,
# stateful Python execution environment ... It is sandboxed to prevent bulk
# file scanning (e.g., glob, os.listdir, os.walk) over the corpus."
# This is load-bearing: it forces the agent to *search* rather than enumerate
# 697 documents. The server below runs in its own process (per-question venv
# when isolation is on), exec()s code into one persistent namespace, and
# captures stdout/stderr at the fd level so subprocess output is seen too.
REPL_SERVER_SOURCE = textwrap.dedent(
    """
    import builtins, glob as _glob, json, os, pathlib, subprocess, sys, traceback

    _BLOCK_MSG = ("Bulk directory enumeration is disabled in this sandbox (OfficeQA Pro, arXiv:2603.08655 §4.1). "
                  "Use the fs_search tool to find files by name or content.")

    def _blocked(*a, **k):
        raise PermissionError(_BLOCK_MSG)

    for _name in ("listdir", "walk", "fwalk", "scandir"):
        setattr(os, _name, _blocked)
    for _name in ("glob", "iglob", "glob0", "glob1"):
        if hasattr(_glob, _name):
            setattr(_glob, _name, _blocked)
    for _name in ("glob", "rglob", "iterdir", "walk"):
        if hasattr(pathlib.Path, _name):
            setattr(pathlib.Path, _name, _blocked)

    _BLOCKED_CMDS = {"ls", "find", "tree", "fd", "fdfind", "locate", "dir"}
    _orig_popen_init = subprocess.Popen.__init__

    def _guarded_popen_init(self, args, *a, **k):
        cmd = args
        if isinstance(args, (list, tuple)) and args:
            cmd = str(args[0])
        elif isinstance(args, (str, bytes)):
            cmd = args.decode() if isinstance(args, bytes) else args
        first = cmd.strip().split()[0] if cmd and cmd.strip() else ""
        if os.path.basename(first) in _BLOCKED_CMDS:
            raise PermissionError(_BLOCK_MSG)
        _orig_popen_init(self, args, *a, **k)

    subprocess.Popen.__init__ = _guarded_popen_init

    _NS = {"__name__": "__main__", "__builtins__": builtins}
    _resp = os.fdopen(int(sys.argv[1]), "w", buffering=1)
    _CAPTURE_PATH = sys.argv[2]

    def _run(code):
        with open(_CAPTURE_PATH, "w+b") as cap:
            saved = (os.dup(1), os.dup(2))
            sys.stdout.flush(); sys.stderr.flush()
            os.dup2(cap.fileno(), 1); os.dup2(cap.fileno(), 2)
            ok = True
            try:
                # Bare expressions echo their value, REPL-style.
                try:
                    compiled = compile(code, "<python_exec>", "eval")
                except SyntaxError:
                    compiled = None
                if compiled is None:
                    exec(compile(code, "<python_exec>", "exec"), _NS)
                else:
                    value = eval(compiled, _NS)
                    if value is not None:
                        print(repr(value))
            except SystemExit:
                pass
            except BaseException:
                ok = False
                traceback.print_exc()
            finally:
                sys.stdout.flush(); sys.stderr.flush()
                os.dup2(saved[0], 1); os.dup2(saved[1], 2)
                os.close(saved[0]); os.close(saved[1])
            cap.seek(0)
            out = cap.read().decode("utf-8", errors="replace")
        return ok, out

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        if req.get("op") == "ping":
            _resp.write(json.dumps({"id": req["id"], "ok": True, "output": "pong"}) + "\\n")
            continue
        ok, out = _run(req["code"])
        _resp.write(json.dumps({"id": req["id"], "ok": ok, "output": out}) + "\\n")
    """
)


class PythonRepl:
    """One persistent Python process per question; ``run`` execs code into a shared namespace."""

    def __init__(
        self, workspace: Workspace, *, default_timeout_s: float = 120.0, max_timeout_s: float = 900.0
    ) -> None:
        self._ws = workspace
        self._default_timeout = default_timeout_s
        self._max_timeout = max_timeout_s
        self._proc: subprocess.Popen[bytes] | None = None
        self._resp_r: Any = None
        self._next_id = 0
        self._lock = asyncio.Lock()

    def _start(self) -> None:
        r, w = os.pipe()
        os.set_inheritable(w, True)
        server = self._ws.root / "_repl_server.py"
        server.write_text(REPL_SERVER_SOURCE, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        self._proc = subprocess.Popen(
            [self._ws.python, "-u", str(server), str(w), str(self._ws.root / "_repl_output.bin")],
            cwd=str(self._ws.cwd),
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            pass_fds=(w,),
            env=env,
        )
        os.close(w)
        self._resp_r = os.fdopen(r, "r", buffering=1)

    def _ensure(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            self._start()

    def _kill(self) -> None:
        if self._proc is not None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=5)
            except Exception:
                pass
        if self._resp_r is not None:
            try:
                self._resp_r.close()
            except Exception:
                pass
        self._proc = None
        self._resp_r = None

    def close(self) -> None:
        self._kill()

    def _roundtrip(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        assert self._proc is not None and self._proc.stdin is not None and self._resp_r is not None
        self._proc.stdin.write((json.dumps(payload) + "\n").encode())
        self._proc.stdin.flush()
        ready, _, _ = select.select([self._resp_r], [], [], timeout)
        if not ready:
            raise TimeoutError
        line = self._resp_r.readline()
        if not line:
            raise RuntimeError("REPL process exited unexpectedly")
        result: dict[str, Any] = json.loads(line)
        return result

    async def run(self, code: str, timeout_s: float | None = None) -> str:
        timeout = min(float(timeout_s or self._default_timeout), self._max_timeout)
        async with self._lock:
            self._ensure()
            self._next_id += 1
            payload = {"id": self._next_id, "code": code}
            try:
                result = await asyncio.to_thread(self._roundtrip, payload, timeout)
            except TimeoutError:
                self._kill()
                return (
                    f"Error: execution exceeded {timeout:.0f}s and the Python process was killed. "
                    "All REPL state (variables, imports) was lost; re-run any setup you need."
                )
            except RuntimeError as e:
                self._kill()
                return f"Error: {e}. All REPL state was lost."
        out = str(result.get("output", ""))
        if not out.strip():
            out = "(no output)" if result.get("ok") else "(error with no output)"
        return out


def make_python_exec(repl: PythonRepl) -> Tool:
    async def python_exec(code: str, timeout: int | None = None) -> str:
        return await repl.run(code, timeout_s=float(timeout) if timeout else None)

    return Tool(
        name="python_exec",
        description=(
            "Execute Python in a persistent, stateful REPL (variables and imports survive between calls). "
            "stdout/stderr are returned. Use it for PDF text extraction / OCR (pymupdf as `fitz`, pdfplumber, "
            "pypdf, pdf2image + pytesseract), numeric computation, and data manipulation. Directory enumeration "
            "(glob, os.listdir, os.walk, ls/find) is disabled; locate files with fs_search instead. "
            "Files are relative to your working directory; the corpus is at ../officeqa_corpus/."
        ),
        parameters={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python source to execute."},
                "timeout": {
                    "type": "integer",
                    "description": "Seconds before the run is killed (default 120, max 900).",
                },
            },
            "required": ["code"],
        },
        fn=python_exec,
    )


# --- web_search -------------------------------------------------------------


class WebSearchBackend(Protocol):
    def search(self, query: str, max_results: int) -> list[dict[str, str]]: ...


class DuckDuckGoBackend:
    """Report §4.1: "A web search tool using the DuckDuckGo Search API". Rate-limited and flaky; swappable."""

    def search(self, query: str, max_results: int) -> list[dict[str, str]]:
        from ddgs import DDGS

        results = DDGS().text(query, max_results=max_results)
        return [
            {"title": str(r.get("title", "")), "url": str(r.get("href", "")), "snippet": str(r.get("body", ""))}
            for r in results or []
        ]


def make_web_search(backend: WebSearchBackend | None = None) -> Tool:
    be = backend or DuckDuckGoBackend()

    async def web_search(query: str, max_results: int = 5) -> str:
        try:
            results = await asyncio.to_thread(be.search, query, max(1, min(int(max_results), 10)))
        except Exception as e:  # network / rate limit
            return f"Error: web search failed ({type(e).__name__}: {e}). Try again later or rephrase."
        if not results:
            return "(no results)"
        return "\n\n".join(f"[{i + 1}] {r['title']}\n{r['url']}\n{r['snippet']}" for i, r in enumerate(results))

    return Tool(
        name="web_search",
        description="Search the open web (DuckDuckGo) for supplementary information, e.g. external statistics.",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "max_results": {"type": "integer", "description": "1-10, default 5."},
            },
            "required": ["query"],
        },
        fn=web_search,
    )


# --- Registry ---------------------------------------------------------------


@dataclass
class ToolSet:
    tools: dict[str, Tool] = field(default_factory=dict)
    repl: PythonRepl | None = None

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self.tools.values()]

    async def call(self, name: str, arguments: dict[str, Any]) -> str:
        tool = self.tools.get(name)
        if tool is None:
            return f"Error: unknown tool {name!r}. Available: {', '.join(self.tools)}."
        try:
            return await tool.fn(**arguments)
        except TypeError as e:
            return f"Error: bad arguments for {name}: {e}"
        except Exception as e:
            logger.exception("tool %s failed", name)
            return f"Error: {type(e).__name__}: {e}"

    def close(self) -> None:
        if self.repl is not None:
            self.repl.close()


def build_toolset(
    workspace: Workspace,
    tools: Sequence[str],
    *,
    output_limit: int = config.TOOL_OUTPUT_LIMIT,
    web_backend: WebSearchBackend | None = None,
) -> ToolSet:
    ts = ToolSet()
    if "fs" in tools:
        for t in (make_fs_search(workspace), make_fs_read(workspace, output_limit=output_limit)):
            ts.tools[t.name] = t
    if "repl" in tools:
        ts.repl = PythonRepl(workspace)
        t = make_python_exec(ts.repl)
        ts.tools[t.name] = t
    if "web" in tools:
        t = make_web_search(web_backend)
        ts.tools[t.name] = t
    return ts
