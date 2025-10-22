from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from app.api.deps import get_current_active_user
from app.models.user import User
from app.agents.orchestrator import create_orchestrator_agent
from app.services.embedding_service import get_embedding_service


router = APIRouter(prefix="/chat", tags=["chat"])

# Create the orchestrator agent once at module level
orchestrator_agent = create_orchestrator_agent()


class ChatRequest(BaseModel):
    message: str
    image_url: Optional[str] = None  # For image-based queries


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
    Enhanced chat endpoint with image support and semantic search.
    
    Examples:
    - Text query: {"message": "Show me black wayfarer glasses"}
    - Image query: {"message": "Similar to this", "image_url": "https://..."}
    - Hybrid: {"message": "Like this but in blue", "image_url": "https://..."}
    """
    if not request.message or not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Build enhanced message with image context
    user_message = request.message
    if request.image_url:
        user_message += f"\n[User provided image URL: {request.image_url}]"
    
    messages = [{"role": "user", "content": user_message}]
    
    # Collect agent response
    all_chunks = []
    tool_called = None
    final_content = None
    
    try:
        config = {"recursion_limit": 10}
        for chunk in orchestrator_agent.stream(
            {"messages": messages}, 
            config=config,
            stream_mode="updates"
        ):
            all_chunks.append(chunk)
            
            if isinstance(chunk, dict):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, dict) and "messages" in node_output:
                        msgs = node_output["messages"]
                        if not isinstance(msgs, list):
                            msgs = [msgs]
                        
                        for msg in msgs:
                            # Extract tool calls
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
        if "recursion" in str(e).lower() and tool_called:
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
    
    # Extract best final content
    if not final_content and all_chunks:
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


@router.get("/search/text")
async def search_text(
    query: str,
    top_k: int = 5,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """
    Direct text search endpoint (bypasses agent).
    Useful for integrations or testing.
    
    Query params:
    - query: search text
    - top_k: number of results (default 5)
    """
    try:
        embedding_service = get_embedding_service()
        results = embedding_service.search_by_text(query, top_k=top_k)
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.get("/search/visual")
async def search_visual(
    image_url: str,
    text_query: Optional[str] = None,
    top_k: int = 5,
    text_weight: float = 0.5,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """
    Direct visual search endpoint (bypasses agent).
    
    Query params:
    - image_url: URL of reference image
    - text_query: optional text refinement
    - top_k: number of results (default 5)
    - text_weight: weight for text vs image (0-1, default 0.5)
    """
    try:
        embedding_service = get_embedding_service()
        results = embedding_service.search_by_image_and_text(
            image_url=image_url,
            text_query=text_query,
            top_k=top_k,
            text_weight=text_weight
        )
        
        return {
            "status": "success",
            "image_url": image_url,
            "text_query": text_query,
            "results_count": len(results),
            "results": results
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Visual search failed: {str(e)}"
        )