from typing import Optional

from app.core.database import get_db_connection
from app.core.security import verify_password
from app.models.user import UserInDB


def get_user(identifier: str) -> Optional[UserInDB]:
    """Retrieve a user by username or email."""
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id, username, email, full_name, password_hash, is_active "
            "FROM users WHERE username = ? OR email = ? LIMIT 1",
            (identifier, identifier),
        )
        row = cur.fetchone()
        if not row:
            return None
        return UserInDB(
            user_id=str(row["user_id"]),
            username=row["username"],
            email=row["email"],
            full_name=row["full_name"],
            hashed_password=row["password_hash"],
            is_active=bool(row["is_active"]),
        )


def authenticate_user(identifier: str, password: str) -> Optional[UserInDB]:
    """Authenticate a user by username/email and password."""
    user = get_user(identifier)
    if not user:
        return None
    if not verify_password(password, user.hashed_password):
        return None
    return user