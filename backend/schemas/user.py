from pydantic import BaseModel, EmailStr
from datetime import datetime

# 1. Base Schema: Properties shared across multiple schemas
class UserBase(BaseModel):
    email: EmailStr

# 2. Create Schema: What the frontend sends us when registering
class UserCreate(UserBase):
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
# 3. Response Schema: What we send back to the frontend (NO PASSWORD)
class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    # This tells Pydantic to read data even if it's not a standard dictionary 
    # (SQLAlchemy objects are class instances, not dicts)
    class Config:
        from_attributes = True