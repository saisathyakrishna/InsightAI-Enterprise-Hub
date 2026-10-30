from sqlalchemy import String, Boolean, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from datetime import datetime
from db.session import Base

class User(Base):
    __tablename__ = "users"

    # We use Mapped[type] for type hinting, and mapped_column for DB constraints
    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # func.now() lets the database handle the exact timestamp automatically
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.now())

    # We will define the relationship to Projects shortly
    # projects: Mapped[list["Project"]] = relationship(back_populates="owner")