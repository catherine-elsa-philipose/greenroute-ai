from fastapi import APIRouter
from app.core.config import settings
from app.schemas.health import HealthResponse

router = APIRouter(tags=["Health"])

@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        version=settings.VERSION
    )
