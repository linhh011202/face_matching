from typing import Optional

from sqlmodel import Field

from app.model.base_model import BaseModel


class UserModel(BaseModel, table=True):
    __tablename__ = "tb_users"

    email: Optional[str] = Field(default=None, index=True, unique=True)
    phone_number: Optional[str] = Field(default=None, unique=True)
    full_name: Optional[str] = Field(default=None)
    password_hashed: Optional[str] = Field(default=None)
    is_ekyc_uploaded: bool = Field(default=False)
