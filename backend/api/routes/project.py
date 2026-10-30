from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
from db.session import get_db
from schemas.project import ProjectCreate, ProjectResponse
from services import project_service
from core.security import get_current_user
from models.user import User

router = APIRouter(prefix="/projects", tags=["Projects"])

@router.post("/", response_model=ProjectResponse)
def create_project(
    project: ProjectCreate, 
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    return project_service.create_user_project(db=db, project=project, user_id=current_user.id)

@router.get("/", response_model=List[ProjectResponse])
def read_projects(
    db: Session = Depends(get_db), 
    current_user: User = Depends(get_current_user)
):
    return project_service.get_user_projects(db=db, user_id=current_user.id)
