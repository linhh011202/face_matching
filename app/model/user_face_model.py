import uuid
from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import ARRAY, BigInteger, Column, DateTime, Text
from sqlmodel import Field, SQLModel, func


class UserFaceModel(SQLModel, table=True):
    __tablename__ = "tb_user_faces"

    id: Optional[int] = Field(
        default=None,
        sa_column=Column(BigInteger, primary_key=True, autoincrement=True),
    )
    user_id: uuid.UUID = Field(nullable=False, index=True)

    embedding: Optional[list[float]] = Field(
        default=None,
        sa_column=Column(Vector(512), nullable=True),
    )
    pose: Optional[str] = Field(default=None)
    source_images: Optional[list[str]] = Field(
        default=None,
        sa_column=Column(ARRAY(Text), nullable=True),
    )

    created_at: datetime = Field(
        sa_column=Column(DateTime(timezone=True), server_default=func.now())
    )
    updated_at: datetime = Field(
        sa_column=Column(
            DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
        )
    )
