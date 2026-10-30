from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from db.session import get_db
from services import rag_service
from core.security import get_current_user
from models.user import User

router = APIRouter(prefix="/rag", tags=["RAG Engine"])

class IngestRequest(BaseModel):
    doc_id: str
    text: str
    project_id: int

class QueryRequest(BaseModel):
    prompt: str
    project_id: int

@router.post("/ingest")
def ingest_doc(data: IngestRequest, current_user: User = Depends(get_current_user)):
    rag_service.ingest_document(doc_id=data.doc_id, text=data.text, project_id=data.project_id)
    return {"status": "success", "message": "Document indexed successfully"}

@router.post("/query")
def query_rag_endpoint(data: QueryRequest, current_user: User = Depends(get_current_user)):
    answer = rag_service.query_rag(prompt=data.prompt, project_id=data.project_id)
    return {"query": data.prompt, "response": answer}