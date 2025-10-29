from typing import Annotated, Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage

from app.api.deps import get_current_active_user
from app.models.user import User
from app.models.conversation import (
    Conversation, ConversationCreate, ConversationWithMessages,
    ConversationSummary
)
from app.models.enhanced_models import ConversationWithProductMessages
from app.crud import conversation as conversation_crud
from app.agents.orchestrator import create_orchestrator_agent
from app.services.embedding_service import get_embedding_service
from app.services.gcs_service import get_gcs_service
import json

router = APIRouter(prefix="/chat", tags=["chat"])

# Create the orchestrator agent once at module level
orchestrator_agent = create_orchestrator_agent()


class ProductResult(BaseModel):
    product_id: str
    name: str
    brand: str
    price: str
    frame_type: Optional[str] = None
    frame_shape: Optional[str] = None
    color: Optional[str] = None
    rating: Optional[float] = None
    reviews: Optional[int] = None
    image_url: Optional[str] = None
    match_score: Optional[str] = None
    similarity_score: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    products: Optional[List[ProductResult]] = None
    tool_called: str
    status: str
    uploaded_image_url: Optional[str] = None
    query: Optional[str] = None
    results_count: Optional[int] = None
    conversation_id: str
    message_count: int


def extract_products_from_tool_response(tool_response: str) -> Optional[List[Dict[str, Any]]]:
    """Extract structured product data from tool JSON response."""
    try:
        import json
        data = json.loads(tool_response)
        
        if data.get("status") == "success" and "products" in data:
            return data["products"]
        
        return None
    except Exception as e:
        print(f"⚠️  Could not parse tool response: {e}")
        return None


# ========== CONVERSATION/SESSION MANAGEMENT ==========

@router.post("/conversations", response_model=Conversation)
async def create_conversation(
    title: Optional[str] = None,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """
    Create a new conversation session.
    Similar to creating a new document in Firestore.
    """
    conversation = ConversationCreate(
        user_id=current_user.user_id,
        title=title
    )
    return conversation_crud.create_conversation(conversation)


@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = Query(50, ge=1, le=100, description="Number of conversations to return"),
    offset: int = Query(0, ge=0, description="Offset for pagination"),
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """
    List all conversation sessions for the current user.
    Returns summaries with timestamps and message previews.
    
    Similar to listing all documents in a Firestore collection.
    """
    return conversation_crud.list_user_conversations(
        user_id=current_user.user_id,
        limit=limit,
        offset=offset
    )


@router.get("/conversations/{conversation_id}", response_model=ConversationWithMessages)
async def get_conversation(
    conversation_id: str,
    limit: Optional[int] = Query(None, description="Limit number of messages returned"),
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """
    Get a specific conversation session with full message history.
    
    Similar to fetching a Firestore document with its subcollection.
    """
    conversation = conversation_crud.get_conversation_with_messages(
        conversation_id=conversation_id,
        user_id=current_user.user_id,
        limit=limit
    )
    
    if not conversation:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )
    
    return conversation


@router.patch("/conversations/{conversation_id}")
async def update_conversation_title(
    conversation_id: str,
    title: str,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """Update the title of a conversation session."""
    success = conversation_crud.update_conversation_title(
        conversation_id=conversation_id,
        user_id=current_user.user_id,
        title=title
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )
    
    return {"status": "success", "message": "Title updated"}


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """Delete a conversation session and all its messages."""
    success = conversation_crud.delete_conversation(
        conversation_id=conversation_id,
        user_id=current_user.user_id
    )
    
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Conversation not found"
        )
    
    return {"status": "success", "message": "Conversation deleted"}


@router.get("/conversations/stats/summary")
async def get_conversation_stats(
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """Get statistics about user's conversations."""
    total_conversations = conversation_crud.get_user_conversation_count(current_user.user_id)
    
    return {
        "user_id": current_user.user_id,
        "total_conversations": total_conversations
    }


# ========== CHAT ENDPOINT (WITHIN SESSIONS) ==========

@router.post("", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    conversation_id: Optional[str] = Form(None),
    image: UploadFile | str | None = File(None),
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """
    Send a message within a conversation session.
    
    Workflow:
    1. If conversation_id provided: Continue existing session
    2. If no conversation_id: Create new session automatically
    3. Add user message to session
    4. Get AI response with conversation history
    5. Add assistant response to session
    
    Parameters:
    - message: User's text query (required)
    - conversation_id: Optional session ID to continue conversation
    - image: Optional image file for visual search
    """
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Get or create conversation session
    if conversation_id:
        conversation = conversation_crud.get_conversation(
            conversation_id=conversation_id,
            user_id=current_user.user_id
        )
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
    else:
        # Auto-create new session
        conversation = conversation_crud.create_conversation(
            ConversationCreate(
                user_id=current_user.user_id,
                title=None  # Will use default timestamp-based title
            )
        )
        conversation_id = conversation.conversation_id
    
    uploaded_image_url = None
    
    # Handle image upload if provided
    if isinstance(image, str) and image == "":
        image = None

    if image and isinstance(image, UploadFile):
        try:
            gcs_service = get_gcs_service()
            uploaded_image_url = gcs_service.upload_user_image(
                file=image,
                user_id=current_user.user_id
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Image upload failed: {str(e)}"
            )
    elif image and not isinstance(image, UploadFile):
        image = None
    
    # Save user message to session
    conversation_crud.add_message(
        conversation_id=conversation_id,
        user_id=current_user.user_id,
        role="user",
        content=message,
        uploaded_image_url=uploaded_image_url
    )
    
    # Get conversation history for context (last 10 messages)
    conversation_with_messages = conversation_crud.get_conversation_with_messages(
        conversation_id=conversation_id,
        user_id=current_user.user_id,
        limit=10
    )
    
    agent_messages = []
    for msg in conversation_with_messages.messages[:-1]:  # Exclude the just-added user message
        if msg.role == "user":
            agent_messages.append(HumanMessage(content=msg.content))
        elif msg.role == "assistant":
            agent_messages.append(AIMessage(content=msg.content))
    
    # Add current message with image context if provided
    user_message = message
    if uploaded_image_url:
        user_message += f"\n[User provided image URL: {uploaded_image_url}]"
    
    agent_messages.append(HumanMessage(content=user_message))
    
    # Call agent
    all_chunks = []
    tool_called = None
    final_content = None
    tool_response_raw = None

    try:
        config = {"recursion_limit": 15}
        
        for chunk in orchestrator_agent.stream(
            {"messages": agent_messages}, 
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
                            
                            # Capture tool response (for product search tools only)
                            if tool_called in ["search_products", "visual_search_products"]:
                                try:
                                    if hasattr(msg, "name") and msg.name in ["search_products", "visual_search_products"]:
                                        if hasattr(msg, "content"):
                                            tool_response_raw = msg.content
                                    elif isinstance(msg, dict) and msg.get("name") in ["search_products", "visual_search_products"]:
                                        tool_response_raw = msg.get("content")
                                except (AttributeError, KeyError):
                                    pass
                            
                            # Capture final content (agent's response)
                            # Key fix: Check for AIMessage type (LangChain's assistant message)
                            try:
                                from langchain_core.messages import AIMessage as LangChainAIMessage
                                
                                if isinstance(msg, LangChainAIMessage):
                                    # It's a LangChain AIMessage
                                    if msg.content and not msg.tool_calls:
                                        final_content = msg.content
                                elif hasattr(msg, "content") and msg.content and hasattr(msg, "role") and msg.role == "assistant":
                                    # Fallback: check role attribute
                                    if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                                        final_content = msg.content
                                elif isinstance(msg, dict) and msg.get("content") and msg.get("role") == "assistant":
                                    # Fallback: dict format
                                    if not msg.get("tool_calls"):
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

    if not tool_called:
        raise HTTPException(
            status_code=500,
            detail="System error: Agent failed to use a required tool."
        )

    # Try to extract final content from last chunk if not found
    if not final_content and all_chunks:
        last_chunk = all_chunks[-1]
        if isinstance(last_chunk, dict):
            for node_output in last_chunk.values():
                if isinstance(node_output, dict) and "messages" in node_output:
                    msgs = node_output["messages"]
                    if not isinstance(msgs, list):
                        msgs = [msgs]
                    
                    for msg in reversed(msgs):
                        try:
                            # Check for AIMessage type first (most reliable)
                            if isinstance(msg, LangChainAIMessage):
                                if msg.content and not msg.tool_calls:
                                    final_content = msg.content
                                    break
                            # Fallback checks
                            elif hasattr(msg, "content") and msg.content and hasattr(msg, "role") and msg.role == "assistant":
                                if not hasattr(msg, "tool_calls") or not msg.tool_calls:
                                    final_content = msg.content
                                    break
                            elif isinstance(msg, dict) and msg.get("content") and msg.get("role") == "assistant":
                                if not msg.get("tool_calls"):
                                    final_content = msg["content"]
                                    break
                        except (AttributeError, KeyError):
                            pass
                
                if final_content:
                    break

    # Handle different tool types
    products = None
    results_count = None
    query = message
    products_json = None

    if tool_called == "handle_conversation":
        # Conversational tool - agent should have generated natural response
        assistant_response = final_content or "I'm here to help! What would you like to know?"
        
    elif tool_called in ["search_products", "visual_search_products"]:
        # Product search - extract products and use agent's formatted response
        if tool_response_raw:
            extracted = extract_products_from_tool_response(tool_response_raw)
            if extracted:
                products = [ProductResult(**p) for p in extracted]
                results_count = len(products)
                products_json = json.dumps(extracted)
                
                try:
                    data = json.loads(tool_response_raw)
                    query = data.get("query") or message
                except:
                    pass
        
        assistant_response = final_content or "I found some products for you."
        
    elif tool_called == "get_order_details":
        # Order tool
        assistant_response = final_content or "Let me check your order details."
        
    else:
        assistant_response = final_content or f"Tool {tool_called} was executed successfully"

    # Save assistant message
    conversation_crud.add_message(
        conversation_id=conversation_id,
        user_id=current_user.user_id,
        role="assistant",
        content=assistant_response,
        tool_called=tool_called,
        products_data=products_json
    )
    
    # Get updated message count
    updated_conversation = conversation_crud.get_conversation(
        conversation_id=conversation_id,
        user_id=current_user.user_id
    )
    
    return ChatResponse(
        response=assistant_response,
        products=products,
        tool_called=tool_called,
        status="success",
        uploaded_image_url=uploaded_image_url,
        query=query or message,
        results_count=results_count,
        conversation_id=conversation_id,
        message_count=updated_conversation.message_count
    )


# ========== DIRECT SEARCH ENDPOINTS (NO SESSIONS) ==========

@router.get("/search/text")
async def search_text(
    query: str,
    top_k: int = 5,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """Direct text search endpoint (bypasses agent and conversation history)."""
    try:
        embedding_service = get_embedding_service()
        results = embedding_service.search_by_text(query, top_k=top_k)
        
        products = [ProductResult(**p) for p in results]
        
        return {
            "status": "success",
            "query": query,
            "results_count": len(products),
            "products": products
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/search/visual")
async def search_visual_upload(
    text_query: Optional[str] = Form(None),
    top_k: int = Form(5),
    text_weight: float = Form(0.5),
    image: UploadFile = File(...),
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """Direct visual search endpoint with file upload (bypasses conversation history)."""
    try:
        gcs_service = get_gcs_service()
        image_url = gcs_service.upload_user_image(
            file=image,
            user_id=current_user.user_id
        )
        
        embedding_service = get_embedding_service()
        results = embedding_service.search_by_image_and_text(
            image_url=image_url,
            text_query=text_query,
            top_k=top_k,
            text_weight=text_weight
        )
        
        products = [ProductResult(**p) for p in results]
        
        return {
            "status": "success",
            "image_url": image_url,
            "text_query": text_query,
            "results_count": len(products),
            "products": products
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Visual search failed: {str(e)}"
        )


# ========== FILE MANAGEMENT ==========

@router.get("/my-uploads")
async def list_my_uploads(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """List all images uploaded by the current user."""
    try:
        gcs_service = get_gcs_service()
        uploads = gcs_service.list_user_files(current_user.user_id)
        
        return {
            "status": "success",
            "user_id": current_user.user_id,
            "upload_count": len(uploads),
            "uploads": uploads
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list uploads: {str(e)}"
        )


@router.delete("/uploads")
async def delete_upload(
    image_url: str,
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """Delete an uploaded image by its public URL."""
    try:
        if f"/{current_user.user_id}/" not in image_url:
            raise HTTPException(
                status_code=403,
                detail="You can only delete your own uploads"
            )
        
        gcs_service = get_gcs_service()
        success = gcs_service.delete_file(image_url)
        
        if success:
            return {
                "status": "success",
                "message": "File deleted successfully",
                "deleted_url": image_url
            }
        else:
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete file: {str(e)}"
        )