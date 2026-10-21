"""Teams (PLAN Phase 8): people who share sandboxes and datasets."""

from datetime import datetime

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base
from app.models.common import created_at_column, prefixed_id


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[str] = mapped_column(String(40), primary_key=True, default=prefixed_id("team"))
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    created_at: Mapped[datetime] = created_at_column()


class TeamMember(Base):
    __tablename__ = "team_members"

    team_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("teams.id", ondelete="CASCADE"), primary_key=True
    )
    user_id: Mapped[str] = mapped_column(
        String(40), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True, index=True
    )
    # "owner" manages members; "member" uses the team's sandboxes and datasets.
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="member")
    created_at: Mapped[datetime] = created_at_column()
