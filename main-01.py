# app_fixed.py
from datetime import datetime, timedelta, timezone
from typing import Optional, Annotated
import os, sqlite3
from contextlib import contextmanager
import uvicorn
import jwt
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from pydantic import BaseModel
from pwdlib import PasswordHash



# LangChain imports
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain.tools import tool

#embeddings



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

# --- LangChain Specialized Agents ---

@tool("order_details", description="Get details about a specific order")
def get_order_details(query: str) -> str:
    """Get details about a specific order based on the user query."""
    # Dummy implementation
    print("I am the order tool, I got a request for:", query)
    return "[ORDER_TOOL] Order details for: {query}"


@tool("product_suggestions", description="Suggest products based on user query")
def suggest_products(query: str) -> str:
    """Suggest products to the user based on their query."""
    # Dummy implementation
    print("I am the product tool, I got a request for:", query)
    return "[PRODUCT_TOOL] Product suggestions for: {query}"


# # Just use plain functions as tools for create_agent
# order_tool = get_order_details
# product_tool = suggest_products


llm = ChatOpenAI(
    model="gpt-3.5-turbo"
)


# Force the agent to always use a tool for every user query
orchestrator_agent = create_agent(
    model=llm,
    tools=[get_order_details, suggest_products],
    system_prompt=(
        "You are a helpful orchestrator that uses tools to answer user queries.\n\n"
        "IMPORTANT RULES:\n"
        "1. For EVERY user query, you MUST call exactly ONE tool before responding\n"
        "2. After the tool returns its result, provide that result to the user\n"
        "3. Tool selection:\n"
        "   - 'get_order_details': for order status, tracking, shipping questions\n"
        "   - 'suggest_products': for product recommendations, browsing, search\n\n"
        "4. WORKFLOW:\n"
        "   - Step 1: Analyze user query\n"
        "   - Step 2: Call the appropriate tool with the query\n"
        "   - Step 3: Return the tool's response to the user\n"
        "   - Step 4: STOP (do not call additional tools)\n\n"
        "5. Pass the complete user query to the tool\n"
        "6. DO NOR ALTER THE RESPONSE FROM THE TOOL IN ANY WAY\n\n"
        "EXAMPLES:\n"
        "User: 'Where is my order?'\n"
        "→ Call get_order_details('Where is my order?')\n"
        "→ Respond with the tool's result\n\n"
        "User: 'Show me laptops'\n"
        "→ Call suggest_products('Show me laptops')\n"
        "→ Respond with the tool's result"
    )
)
# "6. Present the tool's output naturally to the user\n\n"
@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

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
    get_db_connection()

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

@app.post("/chat")
async def chat(message: str, current_user: Annotated[User, Depends(get_current_active_user)]):
    """
    Chat endpoint that enforces tool usage by the orchestrator agent.
    """
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    messages = [{"role": "user", "content": message}]
    
    # Collect all chunks to analyze
    all_chunks = []
    tool_called = None
    final_content = None
    
    try:
        # Stream with recursion limit and collect chunks
        config = {"recursion_limit": 10}  # Reduced limit to catch issues faster
        for chunk in orchestrator_agent.stream(
            {"messages": messages}, 
            config=config,
            stream_mode="updates"
        ):
            all_chunks.append(chunk)
            
            # Extract tool calls as we go
            if isinstance(chunk, dict):
                # Check for tool calls in various node outputs
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, dict) and "messages" in node_output:
                        msgs = node_output["messages"]
                        if not isinstance(msgs, list):
                            msgs = [msgs]
                        
                        for msg in msgs:
                            # Look for tool calls - handle both object and dict formats
                            try:
                                if hasattr(msg, "tool_calls") and msg.tool_calls:
                                    tc = msg.tool_calls[0]
                                    if hasattr(tc, "name"):
                                        tool_called = tc.name
                                    elif isinstance(tc, dict):
                                        tool_called = tc.get("name") or tc.get("function", {}).get("name")
                                elif isinstance(msg, dict) and msg.get("tool_calls"):
                                    tc = msg["tool_calls"][0]
                                    tool_called = tc.get("name") or tc.get("function", {}).get("name")
                            except (IndexError, AttributeError, KeyError) as e:
                                # Skip malformed tool calls
                                pass
                            
                            # Capture final content
                            try:
                                if hasattr(msg, "content") and msg.content:
                                    final_content = msg.content
                                elif isinstance(msg, dict) and msg.get("content"):
                                    final_content = msg["content"]
                            except (AttributeError, KeyError):
                                pass
    
    except Exception as e:
        # If recursion error or other issues, check what we captured
        if "recursion" in str(e).lower() and tool_called:
            # We got at least one tool call before the error
            # This might be OK - the agent called a tool but got stuck in a loop
            pass
        else:
            raise HTTPException(
                status_code=500,
                detail=f"Agent execution error: {str(e)}"
            )
    
    # Verify a tool was called
    if not tool_called:
        raise HTTPException(
            status_code=500,
            detail="System error: Agent failed to use a required tool."
        )
    
    # Extract the best final content
    if not final_content and all_chunks:
        # Try to get content from the last chunk
        last_chunk = all_chunks[-1]
        if isinstance(last_chunk, dict):
            for node_output in last_chunk.values():
                if isinstance(node_output, dict) and "messages" in node_output:
                    msgs = node_output["messages"]
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    for msg in reversed(msgs):
                        if hasattr(msg, "content") and msg.content:
                            final_content = msg.content
                            break
                        elif isinstance(msg, dict) and msg.get("content"):
                            final_content = msg["content"]
                            break
                if final_content:
                    break
    
    return {
        "response": final_content or f"Tool {tool_called} was executed successfully",
        "tool_called": tool_called,
        "status": "success"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)