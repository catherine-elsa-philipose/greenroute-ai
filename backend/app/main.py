from fastapi import FastAPI
from app.core.config import settings
from app.api.v1.router import api_v1_router
from app.schemas.health import HomeResponse, HealthResponse

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API routers
app.include_router(api_v1_router, prefix=settings.API_V1_STR)

# Root endpoint
@app.get("/", response_model=HomeResponse, tags=["Root"])
def home() -> HomeResponse:
    return HomeResponse(
        message="Welcome to GreenRoute AI 🚀"
    )

# Root level health endpoint for backwards compatibility
@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        environment=settings.ENVIRONMENT,
        version=settings.VERSION
    )