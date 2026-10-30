import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.rag_service import ingest_document

logs = [
    {"id": "doc_1", "text": "Major port strike at southern terminals caused a 4-day shipping delay for raw materials.", "project_id": 1},
    {"id": "doc_2", "text": "Supplier XYZ experienced factory equipment failure, reducing component delivery by 20% this month.", "project_id": 1},
    {"id": "doc_3", "text": "Unexpected surge in regional consumer demand due to a viral social media marketing campaign.", "project_id": 1},
    {"id": "doc_4", "text": "Fuel price hikes increased logistics transportation overhead across all major distribution lanes.", "project_id": 1}
]

print("Injecting larger context logs into ChromaDB...")
for log in logs:
    ingest_document(doc_id=log["id"], text=log["text"], project_id=log["project_id"])

print("Successfully loaded bigger context logs into ChromaDB!")