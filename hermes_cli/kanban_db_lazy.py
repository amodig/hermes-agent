"""Lazy access to the Kanban facade during split-module imports."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_UPDATE_UNSET = object()


class _KanbanDbFacade:
    def __getattr__(self, name: str) -> Any:
        return getattr(import_module("hermes_cli.kanban_db"), name)


_kb = _KanbanDbFacade()
