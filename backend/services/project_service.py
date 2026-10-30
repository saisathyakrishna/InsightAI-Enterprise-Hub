from sqlalchemy.orm import Session
from models.project import Project
from schemas.project import ProjectCreate
from models.user import User

def create_user_project(db: Session, project: ProjectCreate, user_id: int):
    db_project = Project(**project.model_dump(), owner_id=user_id)
    db.add(db_project)
    db.commit()
    db.refresh(db_project)
    return db_project

def get_user_projects(db: Session, user_id: int):
    return db.query(Project).filter(Project.owner_id == user_id).all()