from fastapi import APIRouter

from now.bff.v1.routers import search

api_router = APIRouter()
api_router.include_router(search.router, tags=["experiment"])
