from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from app.config import settings

EXEMPT_PATHS = {"/health", "/docs", "/openapi.json", "/redoc"}


class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if settings.api_key and request.url.path not in EXEMPT_PATHS:
            key = request.headers.get("X-API-Key")
            if not key or key != settings.api_key:
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Missing or invalid API key. Provide X-API-Key header."},
                )
        return await call_next(request)
