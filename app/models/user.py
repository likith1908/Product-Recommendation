from typing import Optional
from pydantic import BaseModel


class UserBase(BaseModel):
    username: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = True

class UserCreate(UserBase):
    password: str


class User(UserBase):
    user_id: str
    is_active: int = 1
    
    class Config:
        from_attributes = True


class UserInDB(User):
    hashed_password: str