"""
API Dependencies for authentication and database sessions.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session

from jose import jwt, JWTError, ExpiredSignatureError

from app.database import get_db
from app.config import settings
from app.models import User


# HTTP Bearer token scheme
security = HTTPBearer(auto_error=False)

# JWT constants
ALGORITHM = "HS256"


class TokenPayload:
    """JWT token payload structure."""
    def __init__(
        self,
        sub: str,
        exp: datetime,
        iat: datetime,
        user_id: Optional[str] = None,
        email: Optional[str] = None,
        is_admin: bool = False,
    ):
        self.sub = sub
        self.exp = exp
        self.iat = iat
        self.user_id = user_id
        self.email = email
        self.is_admin = is_admin

    @classmethod
    def from_token(cls, token: str, secret_key: str) -> "TokenPayload":
        """Decode and validate a JWT token."""
        try:
            payload = jwt.decode(token, secret_key, algorithms=[ALGORITHM])
            return cls(
                sub=payload.get("sub", ""),
                exp=datetime.fromtimestamp(payload.get("exp", 0), tz=timezone.utc),
                iat=datetime.fromtimestamp(payload.get("iat", 0), tz=timezone.utc),
                user_id=payload.get("user_id"),
                email=payload.get("email"),
                is_admin=payload.get("is_admin", False),
            )
        except ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Invalid token: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

    def to_dict(self) -> dict:
        """Convert payload to dictionary."""
        return {
            "sub": self.sub,
            "user_id": self.user_id,
            "email": self.email,
            "is_admin": self.is_admin,
        }


def create_access_token(
    data: dict,
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        data: Dictionary containing token payload data
        expires_delta: Optional custom expiration time

    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({
        "exp": expire,
        "iat": datetime.now(timezone.utc),
    })

    encoded_jwt = jwt.encode(
        to_encode,
        settings.secret_key,
        algorithm=ALGORITHM,
    )

    return encoded_jwt


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db),
) -> Optional[dict]:
    """
    Get current user from JWT token.

    Validates the JWT token and returns user information.
    Returns None if no credentials provided (allows anonymous access).
    """
    if not credentials:
        return None

    try:
        # Decode and validate the token
        payload = TokenPayload.from_token(
            credentials.credentials,
            settings.secret_key,
        )

        # Check if token is expired
        if payload.exp < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # If user_id in token, fetch user from database
        if payload.user_id:
            user = db.query(User).filter(User.id == payload.user_id).first()
            if user:
                return {
                    "id": user.id,
                    "email": user.email,
                    "is_admin": user.is_admin,
                    "is_active": user.is_active,
                }

        # Return payload info if no database user found
        return payload.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Authentication failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_auth(
    user: Optional[dict] = Depends(get_current_user),
) -> dict:
    """
    Dependency to require authentication.

    Raises HTTPException if user is not authenticated.
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


def require_admin(
    user: dict = Depends(require_auth),
) -> dict:
    """
    Dependency to require admin privileges.

    Raises HTTPException if user is not an admin.
    """
    if not user.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return user


async def get_db_session() -> Session:
    """
    Wrapper dependency for database session.

    This allows easier testing and mocking.
    """
    for db in get_db():
        yield db
