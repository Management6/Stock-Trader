"""Shared agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class AgentResult:
    agent_name: str
    payload: Any


class BaseAgent(ABC):
    """Minimal common interface for deterministic or future LLM-backed agents."""

    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> AgentResult:
        """Execute the agent's primary task and return an AgentResult."""
