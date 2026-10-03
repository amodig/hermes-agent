"""The bundled review skill is offered in a Kanban worker, not an unrelated session."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.skill_utils import parse_frontmatter, skill_matches_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILL_MD = REPO_ROOT / "skills" / "devops" / "sdlc-review" / "SKILL.md"


@pytest.fixture(scope="module")
def skill_text() -> str:
    return SKILL_MD.read_text(encoding="utf-8")


def test_skill_is_offered_to_kanban_workers(
    skill_text: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    frontmatter, _body = parse_frontmatter(skill_text)
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    monkeypatch.delenv("HERMES_KANBAN_BOARD", raising=False)
    monkeypatch.delenv("HERMES_DELEGATED_CHILD_CONTEXT", raising=False)
    monkeypatch.setattr("tools.kanban_tools._profile_has_kanban_toolset", lambda: False)
    assert skill_matches_environment(frontmatter) is False

    monkeypatch.setenv("HERMES_KANBAN_TASK", "t_0123456789abcdef")
    assert skill_matches_environment(frontmatter) is True
