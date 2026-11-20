"""Teams (PLAN Phase 8): share sandboxes and datasets; see the team's compute usage."""

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel, Field
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_user, not_found
from app.core.errors import ApiError
from app.db.session import get_db
from app.models.sandbox import Sandbox
from app.models.team import Team, TeamMember
from app.models.usage import UsageRecord
from app.models.user import User
from app.schemas.api import Email

router = APIRouter(prefix="/v1/teams", tags=["teams"])


class TeamCreate(BaseModel):
    name: str = Field(min_length=1, max_length=120)


class MemberAdd(BaseModel):
    email: Email
    role: str = Field(default="member", pattern="^(member|owner)$")


async def _membership(db: AsyncSession, team_id: str, user: User) -> TeamMember:
    member = await db.get(TeamMember, (team_id, user.id))
    if member is None:
        raise not_found("Team")
    return member


async def _team_out(db: AsyncSession, team: Team) -> dict:
    rows = await db.execute(
        select(User.id, User.email, TeamMember.role)
        .join(TeamMember, TeamMember.user_id == User.id)
        .where(TeamMember.team_id == team.id)
        .order_by(User.email)
    )
    return {
        "id": team.id,
        "name": team.name,
        "created_at": team.created_at,
        "members": [{"user_id": i, "email": e, "role": r} for i, e, r in rows.all()],
    }


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_team(
    body: TeamCreate, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    team = Team(name=body.name)
    db.add(team)
    await db.flush()
    db.add(TeamMember(team_id=team.id, user_id=user.id, role="owner"))
    await db.commit()
    await db.refresh(team)
    return await _team_out(db, team)


@router.get("")
async def list_teams(
    user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> list[dict]:
    teams = await db.execute(
        select(Team)
        .join(TeamMember, TeamMember.team_id == Team.id)
        .where(TeamMember.user_id == user.id)
        .order_by(Team.name)
    )
    return [await _team_out(db, team) for team in teams.scalars()]


@router.post("/{team_id}/members", status_code=status.HTTP_201_CREATED)
async def add_member(
    team_id: str,
    body: MemberAdd,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    if (await _membership(db, team_id, user)).role != "owner":
        raise ApiError(status.HTTP_403_FORBIDDEN, "not_team_owner", "Only team owners add members")
    invited = await db.scalar(select(User).where(User.email == body.email))
    if invited is None:
        raise not_found("User with this email (they need an account first)")
    existing = await db.get(TeamMember, (team_id, invited.id))
    if existing is None:
        db.add(TeamMember(team_id=team_id, user_id=invited.id, role=body.role))
    else:
        existing.role = body.role
    await db.commit()
    return await _team_out(db, await db.get(Team, team_id))


@router.delete("/{team_id}/members/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def remove_member(
    team_id: str,
    user_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> Response:
    """Owners remove anyone; members can leave."""
    me = await _membership(db, team_id, user)
    if me.role != "owner" and user_id != user.id:
        raise ApiError(
            status.HTTP_403_FORBIDDEN, "not_team_owner", "Only team owners remove members"
        )
    member = await db.get(TeamMember, (team_id, user_id))
    if member is not None:
        await db.delete(member)
        await db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/{team_id}/usage")
async def team_usage(
    team_id: str, user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> dict:
    """Compute used in the team's sandboxes, by member (for team billing)."""
    await _membership(db, team_id, user)
    rows = await db.execute(
        select(User.email, func.sum(UsageRecord.seconds), func.sum(UsageRecord.units))
        .join(Sandbox, Sandbox.id == UsageRecord.sandbox_id)
        .join(User, User.id == UsageRecord.owner_id)
        .where(Sandbox.team_id == team_id)
        .group_by(User.email)
    )
    by_member = {e: {"seconds": round(s, 2), "units": round(u, 2)} for e, s, u in rows.all()}
    return {
        "team_id": team_id,
        "total_units": round(sum(v["units"] for v in by_member.values()), 2),
        "total_seconds": round(sum(v["seconds"] for v in by_member.values()), 2),
        "by_member": by_member,
    }
