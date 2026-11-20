"""Who may use what (PLAN Phase 8): your own things, and your teams' sandboxes and datasets."""

from typing import Any

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.team import TeamMember
from app.models.user import User


async def team_ids(db: AsyncSession, user: User) -> list[str]:
    rows = await db.execute(select(TeamMember.team_id).where(TeamMember.user_id == user.id))
    return list(rows.scalars())


async def visible(db: AsyncSession, user: User, owner_column: Any, team_column: Any) -> Any:
    """A WHERE clause: owned by the user, or shared with one of the user's teams."""
    teams = await team_ids(db, user)
    if not teams:
        return owner_column == user.id
    return or_(owner_column == user.id, team_column.in_(teams))


async def can_use(db: AsyncSession, user: User, owner_id: str, team_id: str | None) -> bool:
    if owner_id == user.id:
        return True
    return team_id is not None and team_id in await team_ids(db, user)
