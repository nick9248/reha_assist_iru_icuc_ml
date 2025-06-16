"""
Production Authentication and Security Middleware - Windows Compatible
Adds API key authentication, rate limiting, and security headers
"""

import os
import time
import hashlib
from typing import Optional
from pathlib import Path
from fastapi import HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import logging

# Import rate limiting with Windows compatibility
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    from slowapi.middleware import SlowAPIMiddleware
    RATE_LIMITING_AVAILABLE = True
except ImportError:
    RATE_LIMITING_AVAILABLE = False

logger = logging.getLogger(__name__)

# Rate limiter configuration (only if available)
if RATE_LIMITING_AVAILABLE:
    limiter = Limiter(key_func=get_remote_address)
else:
    limiter = None

# Security scheme
security = HTTPBearer(auto_error=False)


class APIKeyManager:
    """Manages API key validation for production"""

    def __init__(self):
        # Load environment variables
        self._load_environment()
        # In production, load from environment or secure storage
        self.api_keys = self._load_api_keys()
        self.require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"

    def _load_environment(self):
        """Load environment variables from .env file"""
        try:
            from dotenv import load_dotenv

            # Try project root first
            project_root = Path(__file__).parent.parent.parent
            env_file = project_root / '.env'

            if env_file.exists():
                load_dotenv(env_file)
                logger.info(f"Loaded .env from: {env_file}")
            else:
                # Try current directory
                load_dotenv()
                logger.info("Loaded .env from current directory")

        except ImportError:
            logger.warning("python-dotenv not available, using system environment variables")
        except Exception as e:
            logger.warning(f"Error loading .env file: {e}")

    def _load_api_keys(self) -> dict:
        """Load API keys from environment or configuration"""
        # Default development key (CHANGE IN PRODUCTION!)
        default_keys = {
            "dev-key-12345": {
                "name": "Development Key",
                "permissions": ["predict", "health", "info"],
                "rate_limit": "1000/hour"
            },
            "enaio-integration-key": {
                "name": "enaio DMS Integration",
                "permissions": ["predict", "health"],
                "rate_limit": "5000/hour"
            }
        }

        # Try to load from environment
        env_keys = os.getenv("API_KEYS")
        if env_keys:
            try:
                import json
                return json.loads(env_keys)
            except:
                logger.warning("Failed to parse API_KEYS from environment, using defaults")

        return default_keys

    def validate_api_key(self, api_key: str) -> Optional[dict]:
        """Validate API key and return key info"""
        if not self.require_auth:
            # Development mode - no auth required
            return {"name": "Development Mode", "permissions": ["all"]}

        # Hash the key for secure comparison (in production)
        # For simplicity, using direct comparison here
        if api_key in self.api_keys:
            return self.api_keys[api_key]

        return None

    def check_permission(self, key_info: dict, required_permission: str) -> bool:
        """Check if API key has required permission"""
        if not key_info:
            return False

        permissions = key_info.get("permissions", [])
        return "all" in permissions or required_permission in permissions


# Global API key manager
api_key_manager = APIKeyManager()


async def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    """Verify API key authentication"""

    # Skip auth for health check and docs
    if request.url.path in ["/api/v1/health", "/docs", "/redoc", "/openapi.json", "/"]:
        return {"name": "Public Endpoint", "permissions": ["public"]}

    # Check if auth is required
    if not api_key_manager.require_auth:
        return {"name": "Development Mode", "permissions": ["all"]}

    # Extract API key
    api_key = None
    if credentials:
        api_key = credentials.credentials
    else:
        # Try to get from query parameter (fallback)
        api_key = request.query_params.get("api_key")

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Validate API key
    key_info = api_key_manager.validate_api_key(api_key)
    if not key_info:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Log successful authentication
    logger.info(f"Authenticated request from {key_info['name']}")

    return key_info


async def verify_prediction_permission(
    key_info: dict = Depends(verify_api_key)
) -> dict:
    """Verify permission for prediction endpoints"""

    if not api_key_manager.check_permission(key_info, "predict"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions for prediction endpoints"
        )

    return key_info


class SecurityHeadersMiddleware:
    """Add security headers to responses"""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            async def send_wrapper(message):
                if message["type"] == "http.response.start":
                    headers = dict(message.get("headers", []))

                    # Add security headers
                    security_headers = {
                        b"x-content-type-options": b"nosniff",
                        b"x-frame-options": b"DENY",
                        b"x-xss-protection": b"1; mode=block",
                        b"referrer-policy": b"strict-origin-when-cross-origin"
                    }

                    # Update headers
                    for key, value in security_headers.items():
                        headers[key] = value

                    message["headers"] = list(headers.items())

                await send(message)

            await self.app(scope, receive, send_wrapper)
        else:
            await self.app(scope, receive, send)


# Rate limiting decorators (only if available)
def rate_limit_predictions():
    """Rate limit for prediction endpoints"""
    if RATE_LIMITING_AVAILABLE and limiter:
        return limiter.limit("100/minute")
    else:
        return lambda func: func

def rate_limit_info():
    """Rate limit for info endpoints"""
    if RATE_LIMITING_AVAILABLE and limiter:
        return limiter.limit("60/minute")
    else:
        return lambda func: func

def rate_limit_health():
    """Rate limit for health endpoints"""
    if RATE_LIMITING_AVAILABLE and limiter:
        return limiter.limit("120/minute")
    else:
        return lambda func: func


# Request logging middleware
async def log_requests(request: Request, call_next):
    """Log all requests for monitoring"""
    start_time = time.time()

    # Extract client info
    if RATE_LIMITING_AVAILABLE:
        client_ip = get_remote_address(request)
    else:
        client_ip = request.client.host if request.client else "unknown"

    user_agent = request.headers.get("user-agent", "Unknown")

    # Process request
    response = await call_next(request)

    # Calculate processing time
    process_time = time.time() - start_time

    # Log request
    logger.info(
        f"Request: {request.method} {request.url.path} "
        f"- Status: {response.status_code} "
        f"- Time: {process_time:.3f}s "
        f"- Client: {client_ip} "
        f"- Agent: {user_agent[:50]}"
    )

    # Add process time header
    response.headers["X-Process-Time"] = str(process_time)

    return response


# Environment configuration
class ProductionConfig:
    """Production configuration settings"""

    def __init__(self):
        self.require_auth = os.getenv("REQUIRE_AUTH", "false").lower() == "true"
        self.enable_rate_limiting = os.getenv("ENABLE_RATE_LIMITING", "true").lower() == "true" and RATE_LIMITING_AVAILABLE
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        self.cors_origins = os.getenv("CORS_ORIGINS", "*").split(",")
        self.api_keys = os.getenv("API_KEYS", "")

    def is_production(self) -> bool:
        """Check if running in production mode"""
        env = os.getenv("ENVIRONMENT", "development").lower().strip()
        return env == "production"

    def get_server_config(self) -> dict:
        """Get server configuration"""
        return {
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "workers": int(os.getenv("WORKERS", "4")),
            "timeout": int(os.getenv("TIMEOUT", "30")),
            "keepalive": int(os.getenv("KEEPALIVE", "2"))
        }


# Initialize production config
production_config = ProductionConfig()