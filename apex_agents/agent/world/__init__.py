"""On-disk world surface the APEX-Agents ReAct agent reads.

Each case carries a ``world_id``; the agent navigates a per-case
workspace via :class:`WorldFiles` (zip extraction + temp cleanup with
lazy openpyxl / pypdf / python-docx parsers). Tests bypass HF and use
:class:`FakeWorld` (an in-memory ``{path: content}`` dict that
implements the same surface) so the agent loop can be exercised
without network access.
"""

from .fake import FakeWorld
from .world import WorldFiles, build_world_factory


__all__ = ["FakeWorld", "WorldFiles", "build_world_factory"]
