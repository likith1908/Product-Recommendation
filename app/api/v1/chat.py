from typing import Annotated, Optional, List, Dict, Any

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from app.api.deps import get_current_active_user
from app.models.user import User
from app.agents.orchestrator import create_orchestrator_agent
from app.services.embedding_service import get_embedding_service
from app.services.gcs_service import get_gcs_service


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
    response: str  # Human-readable message
    products: Optional[List[ProductResult]] = None  # Structured product data
    tool_called: str
    status: str
    uploaded_image_url: Optional[str] = None
    query: Optional[str] = None  # Original user query
    results_count: Optional[int] = None


def extract_products_from_tool_response(tool_response: str) -> Optional[List[Dict[str, Any]]]:
    """
    Extract structured product data from tool JSON response.
    
    Args:
        tool_response: JSON string from tool execution
    
    Returns:
        List of product dictionaries or None
    """
    try:
        import json
        data = json.loads(tool_response)
        
        if data.get("status") == "success" and "products" in data:
            return data["products"]
        
        return None
    except Exception as e:
        print(f"⚠️  Could not parse tool response: {e}")
        return None


@router.post("", response_model=ChatResponse)
async def chat(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """
    Enhanced chat endpoint with structured product responses.
    
    Parameters:
    - message: User's text query (required)
    - image: Optional image file for visual search
    
    Response includes:
    - response: Human-readable message
    - products: Array of product objects (if search was performed)
    - tool_called: Which tool was used
    - uploaded_image_url: URL of uploaded image (if any)
    """
    if not message or not message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    uploaded_image_url = None
    
    # Handle image upload if provided
    if image:
        try:
            gcs_service = get_gcs_service()
            uploaded_image_url = gcs_service.upload_user_image(
                file=image,
                user_id=current_user.user_id
            )
            print(f"📸 Image uploaded: {uploaded_image_url}")
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Image upload failed: {str(e)}"
            )
    
    # Build enhanced message with image context
    user_message = message
    if uploaded_image_url:
        user_message += f"\n[User provided image URL: {uploaded_image_url}]"
    
    messages = [{"role": "user", "content": user_message}]
    
    # Collect agent response
    all_chunks = []
    tool_called = None
    final_content = None
    tool_response_raw = None
    
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
                            
                            # Capture tool response (before agent formats it)
                            try:
                                if hasattr(msg, "name") and msg.name in ["search_products", "visual_search_products"]:
                                    if hasattr(msg, "content"):
                                        tool_response_raw = msg.content
                                elif isinstance(msg, dict) and msg.get("name") in ["search_products", "visual_search_products"]:
                                    tool_response_raw = msg.get("content")
                            except (AttributeError, KeyError):
                                pass
                            
                            # Capture final content
                            try:
                                if hasattr(msg, "content") and msg.content and hasattr(msg, "role") and msg.role == "assistant":
                                    final_content = msg.content
                                elif isinstance(msg, dict) and msg.get("content") and msg.get("role") == "assistant":
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
    
    # Extract structured product data
    products = None
    results_count = None
    query = None
    
    if tool_response_raw:
        extracted = extract_products_from_tool_response(tool_response_raw)
        if extracted:
            products = [ProductResult(**p) for p in extracted]
            results_count = len(products)
            
            # Try to extract query from tool response
            try:
                import json
                data = json.loads(tool_response_raw)
                query = data.get("query")
            except:
                pass
    
    return ChatResponse(
        response=final_content or f"Tool {tool_called} was executed successfully",
        products=products,
        tool_called=tool_called,
        status="success",
        uploaded_image_url=uploaded_image_url,
        query=query or message,
        results_count=results_count
    )


@router.get("/search/text")
async def search_text(
    query: str,
    top_k: int = 5,
    current_user: Annotated[User, Depends(get_current_active_user)] = None
):
    """
    Direct text search endpoint (bypasses agent).
    Returns structured product data.
    """
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
    """
    Direct visual search endpoint with file upload.
    Returns structured product data.
    """
    try:
        # Upload image to GCS
        gcs_service = get_gcs_service()
        image_url = gcs_service.upload_user_image(
            file=image,
            user_id=current_user.user_id
        )
        
        # Perform visual search
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


@router.get("/my-uploads")
async def list_my_uploads(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """
    List all images uploaded by the current user.
    """
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
    """
    Delete an uploaded image by its public URL.
    """
    try:
        # Verify the URL belongs to the user's folder
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