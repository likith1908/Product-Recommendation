import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import auth, chat
from app.core.config import settings
from app.core.database import init_db

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="AI-Powered Product Recommendation API with Semantic Search"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    """Initialize database on startup."""
    print("🚀 Starting application...")
    init_db()
    print("✅ Database initialized")
    print("📝 Note: Make sure product embeddings are created!")
    print("   Run: python -m app.scripts.embed_products")


# Include routers
app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(chat.router, prefix=settings.API_V1_PREFIX)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "AI-Powered Product Recommendation API",
        "version": settings.VERSION,
        "features": [
            "Semantic text search with Sentence Transformers",
            "Visual similarity search with CLIP",
            "Hybrid image+text search",
            "Natural language chat interface"
        ],
        "docs": "/docs",
        "endpoints": {
            "auth": f"{settings.API_V1_PREFIX}/auth/token",
            "chat": f"{settings.API_V1_PREFIX}/chat",
            "text_search": f"{settings.API_V1_PREFIX}/chat/search/text",
            "visual_search": f"{settings.API_V1_PREFIX}/chat/search/visual"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "database": settings.DATABASE_PATH,
        "models": {
            "text": settings.SENTENCE_TRANSFORMER_MODEL,
            "image": f"{settings.CLIP_MODEL_NAME} ({settings.CLIP_PRETRAINED})"
        }
    }


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8000)),
        reload=True
    )