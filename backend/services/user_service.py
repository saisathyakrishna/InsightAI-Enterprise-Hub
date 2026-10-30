from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from models.user import User
from schemas.user import UserCreate
from core.security import get_password_hash
from core.security import get_password_hash, verify_password, create_access_token

def authenticate_user(db: Session, email: str, password: str):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def create_user(db: Session, user_data: UserCreate):
    # 1. Check if the email is already registered
    db_user = db.query(User).filter(User.email == user_data.email).first()
    if db_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, 
            detail="Email is already registered"
        )
    
    # 2. Hash the password
    hashed_pw = get_password_hash(user_data.password)
    
    # 3. Create the SQLAlchemy model instance
    new_user = User(
        email=user_data.email, 
        hashed_password=hashed_pw
    )
    
    # 4. Save it to PostgreSQL
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    
    return new_user