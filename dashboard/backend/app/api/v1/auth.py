"""
Authentication API endpoints.

Endpoints:
- POST /login - User login with email and password
- POST /logout - User logout (invalidate token)
- POST /refresh - Refresh access token
- GET /me - Get current user info
"""

from datetime import timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr, Field, field_validator
from sqlalchemy.orm import Session

from app.database import get_db
from app.models import User
from app.schemas import UserResponse
from app.api.deps import get_current_user, create_access_token, TokenPayload
from app.config import settings

# CRITICAL: bcrypt is a HARD dependency for security
# The application must not run without proper password hashing
try:
    import bcrypt
except ImportError:
    raise RuntimeError(
        "CRITICAL: bcrypt is required for password hashing but not installed. "
        "Install with: pip install bcrypt>=4.2.0"
    )

router = APIRouter()
security = HTTPBearer(auto_error=False)


# =============================================================================
# Request/Response schemas
# =============================================================================


class LoginRequest(BaseModel):
    """Login request schema."""
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)

    @field_validator('password')
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets minimum security requirements."""
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number')
        return v


class LoginResponse(BaseModel):
    """Login response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # Token expiration in seconds
    user: UserResponse


class RefreshRequest(BaseModel):
    """Token refresh request schema."""
    refresh_token: str | None = None


class RefreshResponse(BaseModel):
    """Token refresh response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class LogoutResponse(BaseModel):
    """Logout response schema."""
    message: str


# =============================================================================
# Helper functions
# =============================================================================


def verify_password(plain_password: str, hashed_password: str | None) -> bool:
    """
    Verify a password against a hash using bcrypt.

    CRITICAL: This function will raise RuntimeError if bcrypt is not available.
    Never allow password verification without proper hashing.

    Args:
        plain_password: The plain text password to verify
        hashed_password: The bcrypt hash to verify against

    Returns:
        True if password matches, False otherwise
    """
    if not hashed_password:
        return False
    try:
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    except (ValueError, TypeError):
        # Invalid hash format - authentication fails
        return False


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    CRITICAL: This function will raise RuntimeError if bcrypt is not available.
    Never use weak or reversible password hashing.

    Args:
        password: The plain text password to hash

    Returns:
        The bcrypt hash of the password
    """
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def authenticate_user(db: Session, email: str, password: str) -> User | None:
    """
    Authenticate a user by email and password.

    Returns None if authentication fails.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user:
        return None
    if not user.is_active:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user


# =============================================================================
# API endpoints
# =============================================================================


@router.post("/login", response_model=LoginResponse)
async def login(
    login_data: LoginRequest,
    db: Session = Depends(get_db),
):
    """
    User login endpoint.

    Authenticates user with email and password, returns JWT access token.

    **Request:**
    - `email`: User email address
    - `password`: User password (min 8 chars, must contain uppercase, lowercase, and number)

    **Password Requirements:**
    - Minimum 8 characters
    - At least one uppercase letter (A-Z)
    - At least one lowercase letter (a-z)
    - At least one digit (0-9)

    **Response:**
    - `access_token`: JWT access token
    - `token_type`: Token type (bearer)
    - `expires_in`: Token expiration time in seconds
    - `user`: User information

    **Errors:**
    - 401: Invalid email or password
    - 422: Password does not meet requirements
    """
    # Authenticate user
    user = authenticate_user(db, login_data.email, login_data.password)

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={
            "sub": user.email,
            "user_id": str(user.id),
            "email": user.email,
            "is_admin": user.role == "admin",
        },
        expires_delta=access_token_expires,
    )

    return LoginResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            name=user.name,
            role=user.role,
            is_active=user.is_active,
            created_at=user.created_at,
            updated_at=user.updated_at,
        ),
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """
    User logout endpoint.

    Currently, JWT tokens are stateless and cannot be invalidated server-side.
    The client should clear the token from storage.

    In production, consider implementing token blacklist with Redis.
    """
    # For stateless JWT, logout is client-side (clear token)
    # Future enhancement: implement token blacklist
    return LogoutResponse(message="Logout successful. Please clear your token.")


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """
    Refresh access token.

    Returns a new access token for the authenticated user.

    **Note:** Current implementation uses short-lived tokens without refresh tokens.
    For production, implement proper refresh token rotation.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create new access token
    access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
    access_token = create_access_token(
        data={
            "sub": current_user.get("email", ""),
            "user_id": current_user.get("user_id", ""),
            "email": current_user.get("email", ""),
            "is_admin": current_user.get("is_admin", False),
        },
        expires_delta=access_token_expires,
    )

    return RefreshResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=settings.access_token_expire_minutes * 60,
    )


@router.get("/me", response_model=UserResponse)
async def get_me(
    current_user: Annotated[dict, Depends(get_current_user)],
):
    """
    Get current user information.

    Returns the authenticated user's profile.
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserResponse(
        id=current_user.get("user_id", ""),
        email=current_user.get("email", ""),
        name=current_user.get("name", current_user.get("email", "").split("@")[0]),
        role="admin" if current_user.get("is_admin") else "editor",
        is_active=current_user.get("is_active", True),
        created_at=current_user.get("created_at", ""),
        updated_at=current_user.get("updated_at", ""),
    )
