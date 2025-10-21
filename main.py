# app_fixed.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Annotated
import os, sqlite3
from contextlib import contextmanager

import jwt
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel
from pwdlib import PasswordHash

SECRET_KEY = os.environ.get("JWT_SECRET", "changeme")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
DATABASE_PATH = os.environ.get("DATABASE_PATH", "sample_data.db")

class Token(BaseModel):
    access_token: str
    token_type: str

class User(BaseModel):
    username: str
    user_id: Optional[str] = None
    email: Optional[str] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = True

class UserInDB(User):
    hashed_password: Optional[str] = None

password_hash = PasswordHash.recommended()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")
app = FastAPI()

@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db():
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                email TEXT UNIQUE,
                full_name TEXT,
                password_hash TEXT NOT NULL,
                is_active INTEGER DEFAULT 1
            )
        """)
        conn.commit()
        cur.execute("PRAGMA table_info(users)")
        cols = [r[1] for r in cur.fetchall()]
        if "username" in cols and "hashed_password" in cols:
            cur.execute("SELECT 1 FROM users WHERE username = ?", ("johndoe",))
            if not cur.fetchone():
                cur.execute(
                    "INSERT INTO users (username, email, full_name, hashed_password, is_active) VALUES (?, ?, ?, ?, ?)",
                    ("johndoe", "johndoe@example.com", "John Doe",
                     "$argon2id$v=19$m=65536,t=3,p=4$wagCPXjifgvUFBzq4hqe3w$CYaIb8sB+wtD+Vu/P4uod1+Qof8h+1g7bbDlBID48Rc",
                     0),
                )
                conn.commit()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    if not hashed_password:
        return False
    try:
        return password_hash.verify(plain_password, hashed_password)
    except Exception:
        return plain_password == hashed_password

def get_user(identifier: str) -> Optional[UserInDB]:
    with get_db_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            "SELECT user_id, username, email, full_name, password_hash, is_active FROM users WHERE username = ? OR email = ? LIMIT 1",
            (identifier, identifier),
        )
        row = cur.fetchone()
        if not row:
            return None
        return UserInDB(
            user_id=row["user_id"],
            username=row["username"],
            email=row["email"],
            full_name=row["full_name"],
            hashed_password=row["password_hash"],
            is_active=bool(row["is_active"]),
        )

def authenticate_user(identifier: str, password: str) -> Optional[UserInDB]:
    user = get_user(identifier)
    if not user:
        return None
    if not verify_password(password, user.hashed_password or ""):
        return None
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: Annotated[str, Depends(oauth2_scheme)]) -> User:
    cred_exc = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
                             headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise cred_exc
    except InvalidTokenError:
        raise cred_exc
    user = get_user(username)
    if user is None:
        raise cred_exc
    return user

async def get_current_active_user(current_user: Annotated[User, Depends(get_current_user)]):
    if not getattr(current_user, "is_active", True):
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@app.on_event("startup")
def startup_event():
    init_db()

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()]):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"})
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return Token(access_token=access_token, token_type="bearer")

@app.get("/users/me/", response_model=User)
async def read_users_me(current_user: Annotated[User, Depends(get_current_active_user)]):
    return current_user

@app.get("/users/me/items/")
async def read_own_items(current_user: Annotated[User, Depends(get_current_active_user)]):
    return [{"item_id": current_user.user_id, "owner": current_user.username}]
