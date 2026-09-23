from fastapi import APIRouter

from app.api.v1 import articles, health, model, sources

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(model.router)
api_router.include_router(sources.router)
api_router.include_router(articles.router)

__all__ = ["api_router"]
