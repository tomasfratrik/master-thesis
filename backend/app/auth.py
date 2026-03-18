import hashlib
import hmac
import secrets
import uuid
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from .db import get_connection, utc_now


security = HTTPBearer(auto_error=True)


def hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt.encode("utf-8"),
        200_000,
    ).hex()


def normalize_username(username: str) -> str:
    normalized = "".join(character for character in username.strip().lower() if character.isalnum() or character in {"_", "-", "."})
    if not normalized:
        raise HTTPException(status_code=400, detail="Username is required.")
    return normalized


def _fallback_email(username: str) -> str:
    return f"{username}@local"


def _serialize_user(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "username": row["username"],
        "full_name": row["full_name"],
        "email": row["email"],
        "role": row["role"],
    }


def create_user(
    *,
    username: str,
    password: str,
    full_name: str | None = None,
    email: str | None = None,
    role: str = "user",
) -> dict[str, Any]:
    username = normalize_username(username)
    email = (email or _fallback_email(username)).strip().lower()
    full_name = (full_name or username).strip()
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    user_id = str(uuid.uuid4())

    with get_connection() as connection:
        try:
            connection.execute(
                """
                INSERT INTO users (id, email, username, full_name, role, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (user_id, email, username, full_name, role, password_hash, salt, utc_now()),
            )
        except Exception as error:
            raise HTTPException(status_code=409, detail="User already exists.") from error

    return {
        "id": user_id,
        "username": username,
        "full_name": full_name,
        "email": email,
        "role": role,
    }


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    username = normalize_username(username)
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,),
        ).fetchone()

    if row is None:
        return None

    candidate_hash = hash_password(password, row["password_salt"])
    if not hmac.compare_digest(candidate_hash, row["password_hash"]):
        return None

    return _serialize_user(dict(row))


def ensure_demo_users() -> None:
    defaults = (
        {
            "username": "user",
            "password": "user",
            "full_name": "Demo User",
            "email": "user@local",
            "role": "user",
        },
        {
            "username": "admin",
            "password": "admin",
            "full_name": "Admin User",
            "email": "admin@local",
            "role": "admin",
        },
    )

    with get_connection() as connection:
        for item in defaults:
            row = connection.execute(
                "SELECT id FROM users WHERE username = ?",
                (item["username"],),
            ).fetchone()
            if row is not None:
                continue

            salt = secrets.token_hex(16)
            password_hash = hash_password(item["password"], salt)
            connection.execute(
                """
                INSERT INTO users (id, email, username, full_name, role, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    item["email"],
                    item["username"],
                    item["full_name"],
                    item["role"],
                    password_hash,
                    salt,
                    utc_now(),
                ),
            )


def create_session(user_id: str) -> str:
    token = secrets.token_urlsafe(32)
    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO sessions (token, user_id, created_at)
            VALUES (?, ?, ?)
            """,
            (token, user_id, utc_now()),
        )
    return token


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict[str, Any]:
    with get_connection() as connection:
        row = connection.execute(
            """
            SELECT users.*
            FROM sessions
            JOIN users ON users.id = sessions.user_id
            WHERE sessions.token = ?
            """,
            (credentials.credentials,),
        ).fetchone()

    if row is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token.")

    return _serialize_user(dict(row))
