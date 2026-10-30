import chromadb
from chromadb.config import Settings

# Persistent client stores embeddings locally in a 'chroma_db' directory
chroma_client = chromadb.PersistentClient(path="./chroma_db")

def get_vector_collection(name: str = "insightai_docs"):
    return chroma_client.get_or_create_collection(name=name)