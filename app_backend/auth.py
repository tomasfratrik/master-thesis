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


def create_user(email: str, password: str) -> dict[str, Any]:
    email = email.strip().lower()
    salt = secrets.token_hex(16)
    password_hash = hash_password(password, salt)
    user_id = str(uuid.uuid4())

    with get_connection() as connection:
        try:
            connection.execute(
                """
                INSERT INTO users (id, email, password_hash, password_salt, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, email, password_hash, salt, utc_now()),
            )
        except Exception as error:
            raise HTTPException(status_code=409, detail="User already exists.") from error

    return {"id": user_id, "email": email}


def authenticate_user(email: str, password: str) -> dict[str, Any] | None:
    email = email.strip().lower()
    with get_connection() as connection:
        row = connection.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()

    if row is None:
        return None

    candidate_hash = hash_password(password, row["password_salt"])
    if not hmac.compare_digest(candidate_hash, row["password_hash"]):
        return None

    return dict(row)


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

    return dict(row)
