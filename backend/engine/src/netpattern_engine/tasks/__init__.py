"""Task plugins (PLAN §6): what a task's targets, losses, metrics and outputs look like."""

from netpattern_engine.tasks.base import TaskPlugin
from netpattern_engine.tasks.registry import TASKS, get_task

__all__ = ["TASKS", "TaskPlugin", "get_task"]
