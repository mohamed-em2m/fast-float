from fastapi import APIRouter
from fast_float.routers.generation import router as generation_router
from fast_float.routers.health import router as health_router
from fast_float.routers.webrtc import router as webrtc_router

api_router = APIRouter()
api_router.include_router(generation_router)
api_router.include_router(health_router)
api_router.include_router(webrtc_router)

__all__ = ["api_router", "generation_router", "health_router", "webrtc_router"]
