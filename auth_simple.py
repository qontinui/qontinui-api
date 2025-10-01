"""
Simple authentication module for Qontinui API - temporary solution
"""

import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, Form, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# Configuration
SECRET_KEY = "your-secret-key-here-change-in-production"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Simple file-based storage
USERS_FILE = Path(__file__).parent / "users.json"

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# Router
router = APIRouter(prefix="/auth", tags=["authentication"])


# Models
class User(BaseModel):
    id: int
    email: str
    username: str
    full_name: str | None = None
    is_active: bool = True


class UserCreate(BaseModel):
    email: str
    username: str
    password: str
    full_name: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: str | None = None


# Initialize users file if doesn't exist
if not USERS_FILE.exists():
    USERS_FILE.write_text(json.dumps({"users": {}}))


# Helper functions
def hash_password(password: str) -> str:
    """Simple password hashing using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return hash_password(plain_password) == hashed_password


def load_users() -> dict:
    """Load users from file"""
    return json.loads(USERS_FILE.read_text())


def save_users(users_data: dict):
    """Save users to file"""
    USERS_FILE.write_text(json.dumps(users_data, indent=2))


def get_user_by_username(username: str) -> dict | None:
    """Get user by username"""
    users_data = load_users()
    return users_data["users"].get(username)


def authenticate_user(username: str, password: str) -> dict | None:
    """Authenticate a user"""
    user = get_user_by_username(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return user


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """Create an access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire, "type": "access"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def create_refresh_token(data: dict) -> str:
    """Create a refresh token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """Get the current authenticated user"""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError as e:
        raise credentials_exception from e

    user = get_user_by_username(username=token_data.username)
    if user is None:
        raise credentials_exception

    return User(
        id=user["id"],
        email=user["email"],
        username=user["username"],
        full_name=user.get("full_name"),
        is_active=user.get("is_active", True),
    )


# Endpoints
@router.post("/register", response_model=User)
async def register(user_data: UserCreate):
    """Register a new user"""
    users_data = load_users()

    # Check if user exists
    if user_data.username in users_data["users"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Username already taken"
        )

    # Check if email exists
    for u in users_data["users"].values():
        if u["email"] == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered"
            )

    # Create user
    user_id = len(users_data["users"]) + 1
    new_user = {
        "id": user_id,
        "email": user_data.email,
        "username": user_data.username,
        "full_name": user_data.full_name,
        "hashed_password": hash_password(user_data.password),
        "is_active": True,
        "created_at": datetime.now().isoformat(),
        "projects": [],
    }

    users_data["users"][user_data.username] = new_user
    save_users(users_data)

    return User(
        id=user_id,
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        is_active=True,
    )


@router.post("/login", response_model=TokenResponse)
async def login(username: str = Form(...), password: str = Form(...)):
    """Login and get access tokens"""
    user = authenticate_user(username, password)
    if not user:
        # For testing, create a default user if none exists
        users_data = load_users()
        if len(users_data["users"]) == 0:
            # Create default test user
            default_user = {
                "id": 1,
                "email": "test@example.com",
                "username": "test",
                "full_name": "Test User",
                "hashed_password": hash_password("test"),
                "is_active": True,
                "created_at": datetime.now().isoformat(),
                "projects": [],
            }
            users_data["users"]["test"] = default_user
            save_users(users_data)

            # Try authenticating again
            if username == "test" and password == "test":
                user = default_user
            else:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Incorrect username or password. Try username: test, password: test",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

    access_token = create_access_token(data={"sub": user["username"]})
    refresh_token = create_refresh_token(data={"sub": user["username"]})

    return TokenResponse(
        access_token=access_token, refresh_token=refresh_token, token_type="bearer"
    )


@router.get("/me", response_model=User)
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user info"""
    return current_user


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """Refresh access token"""
    try:
        payload = jwt.decode(refresh_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        token_type: str = payload.get("type")

        if username is None or token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
            )

        user = get_user_by_username(username)
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

        new_access_token = create_access_token(data={"sub": user["username"]})
        new_refresh_token = create_refresh_token(data={"sub": user["username"]})

        return TokenResponse(
            access_token=new_access_token, refresh_token=new_refresh_token, token_type="bearer"
        )
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token"
        ) from e


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout (placeholder for token blacklisting)"""
    return {"message": "Successfully logged out"}


# Project management endpoints
@router.get("/projects")
async def get_user_projects(current_user: User = Depends(get_current_user)):
    """Get all projects for the current user"""
    users_data = load_users()
    user = users_data["users"].get(current_user.username)
    if user:
        return {"projects": user.get("projects", [])}
    return {"projects": []}


@router.post("/projects")
async def save_project(
    project_data: dict[str, Any], current_user: User = Depends(get_current_user)
):
    """Save a project for the current user"""
    users_data = load_users()
    user = users_data["users"].get(current_user.username)

    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Get or create projects list
    if "projects" not in user:
        user["projects"] = []

    # Add or update project
    project_id = project_data.get("id")
    if project_id:
        # Update existing project
        user["projects"] = [p for p in user["projects"] if p.get("id") != project_id]
    else:
        # New project
        project_data["id"] = f"project_{datetime.now().timestamp()}"

    project_data["updated_at"] = datetime.now().isoformat()
    user["projects"].append(project_data)

    # Save back
    save_users(users_data)

    return {"message": "Project saved successfully", "project_id": project_data["id"]}
