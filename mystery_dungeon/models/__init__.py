"""Contracts for planning and action models, the configured model table, and loading a model."""

from __future__ import annotations

import importlib
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from emulator import Frame
    from mystery_dungeon.actions import Action
    from mystery_dungeon.agent import MemoryEntry

CONFIG = Path(__file__).parents[2] / "config.toml"


class ActionModel(Protocol):
    """Chooses the next action for a task."""

    last_response: dict | None  # raw answer to the latest call, kept for the run log

    def choose_action(self, task: str, knowledge: str, frame: Frame,
                      memory: tuple[MemoryEntry, ...]) -> Action:
        """Return one action from ACTIONS for the current frame.

        Args:
            memory: recent screenshots, oldest first, each with the action taken after it.

        Raises:
            Exception: when the model fails or answers outside the action space.
        """
        ...


class PlanningModel(Protocol):
    """Judges progress on a task."""

    last_response: dict | None

    def is_complete(self, task: str, frame: Frame, memory: tuple[MemoryEntry, ...]) -> bool:
        """Judge from visible evidence only whether the task has been completed.

        Raises:
            Exception: when the model fails or does not answer yes or no.
        """
        ...


class Model(ActionModel, PlanningModel, Protocol):
    """One model filling both roles."""


@dataclass(frozen=True)
class ModelEntry:
    """One [models.NAME] table: an adapter and either a command that serves it or its URL.

    Without ``serve`` or ``url`` the adapter uses its own default endpoint.
    """

    name: str
    label: str
    spec: str
    serve: tuple[str, ...] = ()
    ready: str = ""
    url: str | None = None
    files: tuple[str, ...] = ()  # local weights the benchmark hashes, relative to config.toml


def read_models(path: Path = CONFIG) -> dict[str, ModelEntry]:
    """Read the [models] tables of a TOML config, in file order.

    Raises:
        TypeError: when a table is missing label or spec, or has an unknown key.
        ValueError: when serve or files is not a list of strings, a table sets both serve
            and url, or serve has no ready path starting with "/".
    """
    tables = tomllib.loads(path.read_text()).get("models", {})
    if not tables:
        raise ValueError(f"{path} defines no [models.NAME] tables")
    entries = {}
    for name, table in tables.items():
        for key in ("serve", "files"):
            value = table.get(key, [])
            if not isinstance(value, list) or not all(isinstance(arg, str) for arg in value):
                raise ValueError(f"models.{name}.{key} must be a list of strings")
        if "serve" in table and "url" in table:
            raise ValueError(f"models.{name} sets both serve and url")
        if "serve" in table and not str(table.get("ready", "")).startswith("/"):
            raise ValueError(f"models.{name} sets serve without a ready path starting with /")
        entries[name] = ModelEntry(name, **{**table, "serve": tuple(table.get("serve", ())),
                                            "files": tuple(table.get("files", ()))})
    return entries


def load_model(spec: str, url: str | None = None) -> Model:
    """Instantiate MODULE:CLASS with the URL of its server, or None for its default.

    Raises:
        ValueError: when spec is not MODULE:CLASS.
        TypeError: when the class does not provide both model roles.
    """
    module, separator, class_name = spec.partition(":")
    if not (module and separator and class_name):
        raise ValueError("model must be MODULE:CLASS")
    cls = getattr(importlib.import_module(module), class_name)
    if not isinstance(cls, type) or not all(
            callable(getattr(cls, method, None)) for method in ("choose_action", "is_complete")):
        raise TypeError(f"{spec} must be a class with choose_action and is_complete")
    return cls(url)
