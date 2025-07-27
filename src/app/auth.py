#!/usr/bin/env python3
"""
JWT-based authentication for FastAPI inference service
"""
import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET = os.getenv(
    "JWT_SECRET", "conviction-ai-super-secret-key-change-in-production"
)
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

# Security scheme
security = HTTPBearer(auto_error=False)


class TokenData(BaseModel):
    """Token payload data"""

    user_id: str
    username: str
    permissions: list = []
    exp: datetime


class AuthError(HTTPException):
    """Custom authentication error"""

    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


def create_access_token(user_id: str, username: str, permissions: list = None) -> str:
    """
    Create a JWT access token

    Args:
        user_id: User identifier
        username: Username
        permissions: List of permissions

    Returns:
        JWT token string
    """
    if permissions is None:
        permissions = ["predict"]

    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)

    payload = {
        "user_id": user_id,
        "username": username,
        "permissions": permissions,
        "exp": expire,
        "iat": datetime.utcnow(),
        "iss": "conviction-ai-inference",
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.info(f"Created token for user {username}")

    return token


def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> TokenData:
    """
    Verify JWT token and return token data

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        TokenData object

    Raises:
        AuthError: If token is invalid or expired
    """
    if not credentials:
        raise AuthError("Missing authentication token")

    try:
        payload = jwt.decode(
            credentials.credentials, JWT_SECRET, algorithms=[JWT_ALGORITHM]
        )

        # Validate required fields
        user_id = payload.get("user_id")
        username = payload.get("username")

        if not user_id or not username:
            raise AuthError("Invalid token payload")

        token_data = TokenData(
            user_id=user_id,
            username=username,
            permissions=payload.get("permissions", []),
            exp=datetime.fromtimestamp(payload.get("exp", 0)),
        )

        logger.debug(f"Token verified for user {username}")
        return token_data

    except jwt.ExpiredSignatureError:
        raise AuthError("Token has expired")
    except jwt.InvalidTokenError as e:
        logger.warning(f"Invalid token: {str(e)}")
        raise AuthError("Invalid token")
    except Exception as e:
        logger.error(f"Token verification error: {str(e)}")
        raise AuthError("Token verification failed")


def verify_permission(required_permission: str):
    """
    Dependency to verify user has required permission

    Args:
        required_permission: Permission string to check

    Returns:
        Dependency function
    """

    def permission_checker(token_data: TokenData = Depends(verify_token)) -> TokenData:
        if required_permission not in token_data.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_permission}",
            )
        return token_data

    return permission_checker


def optional_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> Optional[TokenData]:
    """
    Optional authentication - returns None if no token provided

    Args:
        credentials: HTTP Bearer credentials

    Returns:
        TokenData object or None
    """
    if not credentials:
        return None

    try:
        return verify_token(credentials)
    except AuthError:
        return None


# Pre-defined permission checkers
verify_predict_permission = verify_permission("predict")
verify_admin_permission = verify_permission("admin")
verify_batch_permission = verify_permission("batch")


def generate_test_token(username: str = "test_user", permissions: list = None) -> str:
    """
    Generate a test token for development/testing

    Args:
        username: Test username
        permissions: List of permissions

    Returns:
        JWT token string
    """
    if permissions is None:
        permissions = ["predict", "batch", "admin"]

    return create_access_token(
        user_id="test_123", username=username, permissions=permissions
    )


def get_user_info(token_data: TokenData = Depends(verify_token)) -> Dict[str, Any]:
    """
    Get user information from token

    Args:
        token_data: Verified token data

    Returns:
        User information dictionary
    """
    return {
        "user_id": token_data.user_id,
        "username": token_data.username,
        "permissions": token_data.permissions,
        "token_expires": token_data.exp.isoformat(),
    }
