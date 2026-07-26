"""On-disk world surface the APEX-Agents ReAct agent reads.

Each case carries a ``world_id``; the agent navigates a per-case
workspace via :class:`WorldFiles` (zip extraction + temp cleanup with
lazy openpyxl / pypdf / python-docx parsers). Tests bypass HF with an
in-memory ``FakeWorld`` shim that implements the same read surface so
the agent loop can be exercised without network access; that shim is
test-only and lives in ``apex_agents.tests.fake_world`` (not shipped
in the wheel).
"""

from .world import WorldFiles, build_world_factory


__all__ = ["WorldFiles", "build_world_factory"]
