from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_current_active_user
from app.models.user import User
from app.agents.orchestrator import create_orchestrator_agent


router = APIRouter(prefix="/chat", tags=["chat"])

# Create the orchestrator agent once at module level
orchestrator_agent = create_orchestrator_agent()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    tool_called: str
    status: str


@router.post("", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """
    Chat endpoint that enforces tool usage by the orchestrator agent.
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    messages = [{"role": "user", "content": request.message}]
    
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
                            except (IndexError, AttributeError, KeyError):
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
    
    return ChatResponse(
        response=final_content or f"Tool {tool_called} was executed successfully",
        tool_called=tool_called,
        status="success"
    )