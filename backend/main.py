from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from scalar_fastapi import get_scalar_api_reference
from api.routes import user, project, rag, ml

app = FastAPI(
    title="InsightAI Enterprise API",
    description="Production-grade backend platform for intelligent supply chain management, predictive demand forecasting, and RAG document retrieval.",
    version="1.0.0",
    docs_url=None  # Disable default Swagger UI
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user.router, prefix="/api/v1")
app.include_router(project.router, prefix="/api/v1")
app.include_router(rag.router, prefix="/api/v1")
app.include_router(ml.router, prefix="/api/v1")

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy", "service": "InsightAI"}

# Mount gorgeous Scalar API reference UI
@app.get("/docs", include_in_schema=False)
def scalar_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title="InsightAI Enterprise API",
        theme="mars",  # Options: 'purple', 'solarized', 'mars', 'kepler', 'deepSpace', 'saturn'
    )