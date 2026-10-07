"""Errors the engine reports back to the platform.

`SpecError` carries a list of issues that point at a pipeline node and field, so the
UI can highlight the exact spot (PLAN §9).
"""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Issue:
    message: str
    node_id: str | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


class EngineError(Exception):
    """An error with a message that is safe to show to users."""


class SpecError(EngineError):
    def __init__(self, issues: list[Issue]) -> None:
        self.issues = issues
        summary = "; ".join(_describe(issue) for issue in issues[:5])
        more = f" (+{len(issues) - 5} more)" if len(issues) > 5 else ""
        super().__init__(f"{len(issues)} problem(s) in the spec: {summary}{more}")

    def to_dict(self) -> dict[str, Any]:
        return {"message": str(self), "issues": [issue.to_dict() for issue in self.issues]}


class MissingPackError(EngineError):
    """The pipeline needs a capability pack that this environment does not include."""

    def __init__(self, pack: str, reason: str) -> None:
        self.pack = pack
        super().__init__(
            f"{reason} needs the '{pack}' capability pack, which this environment lacks"
        )


def _describe(issue: Issue) -> str:
    where = ".".join(part for part in (issue.node_id, issue.field) if part)
    return f"{where}: {issue.message}" if where else issue.message


def issues_from_validation_error(
    error: Any, node_id: str | None = None, prefix: str = ""
) -> list[Issue]:
    """Convert a pydantic ValidationError into issues (field paths joined with dots)."""
    issues = []
    for detail in error.errors():
        location = ".".join(str(part) for part in detail["loc"])
        field = ".".join(part for part in (prefix, location) if part) or None
        issues.append(Issue(message=detail["msg"], node_id=node_id, field=field))
    return issues
