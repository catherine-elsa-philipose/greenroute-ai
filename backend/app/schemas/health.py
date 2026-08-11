from pydantic import BaseModel, Field

class HealthResponse(BaseModel):
    status: str = Field(default="healthy", description="Current health status of the application")
    environment: str = Field(..., description="Current environment mode")
    version: str = Field(..., description="Application version")

class HomeResponse(BaseModel):
    message: str = Field(..., description="Welcome message")
