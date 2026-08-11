import os

class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "GreenRoute AI")
    PROJECT_DESCRIPTION: str = os.getenv("PROJECT_DESCRIPTION", "Explainable Confidence-Aware Multi-LLM Router")
    VERSION: str = os.getenv("VERSION", "0.1.0")
    API_V1_STR: str = "/api/v1"
    
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "true").lower() in ("true", "1", "t")
    
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:5174")

settings = Settings()
