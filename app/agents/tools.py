import json
from langchain.tools import tool
from app.services.embedding_service import get_embedding_service

@tool("handle_conversation", description="Handle conversational queries without search/orders")
def handle_conversation(query_type: str, user_message: str) -> str:
    """
    Handle general conversation, greetings, capability questions, and follow-ups.
    
    Args:
        query_type: "greeting", "capability", "clarification", "follow_up", "history_reference"
        user_message: The original user message
    """
    print(f"💬 Conversation tool called")
    print(f"   Type: {query_type}")
    print(f"   Message: {user_message}")
    
    return json.dumps({
        "status": "success",
        "tool": "handle_conversation",
        "query_type": query_type,
        "user_message": user_message,
        "instruction": "Use conversation history to respond naturally and helpfully."
    }, indent=2)

@tool("order_details", description="Get details about a specific order")
def get_order_details(query: str) -> str:
    """Get details about a specific order based on the user query."""
    print(f"🔍 Order tool called with query: {query}")
    
    return json.dumps({
        "status": "success",
        "message": "Order tracking feature - connect to your order database",
        "query": query,
        "note": "This is a placeholder. Implement your order tracking logic here."
    }, indent=2)


@tool("search_products", description="Search and recommend products using semantic understanding")
def search_products(query: str) -> str:
    """
    Search products using semantic embeddings. Understands natural language queries.
    
    Examples:
    - "Show me black wayfarer glasses"
    - "I need reading glasses for a round face"
    - "Lightweight sports sunglasses under 2000 rupees"
    """
    print(f"🔍 Product search tool called with query: {query}")
    
    try:
        embedding_service = get_embedding_service()
        
        # Simple filter extraction from query
        filters = {}
        query_lower = query.lower()
        
        # Brand filtering
        brands = ["john jacobs", "ray-ban", "oakley", "vincent chase"]
        for brand in brands:
            if brand in query_lower:
                # Capitalize properly
                filters["Brand Name"] = brand.title()
                break
        
        # Frame type filtering
        if "full rim" in query_lower:
            filters["Frame Type"] = "Full Rim"
        elif "half rim" in query_lower:
            filters["Frame Type"] = "Half Rim"
        elif "rimless" in query_lower:
            filters["Frame Type"] = "Rimless"
        
        # Search using embeddings
        results = embedding_service.search_by_text(
            query=query,
            top_k=5,
            filters=filters if filters else None
        )
        
        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "No products found matching your query. Try different search terms.",
                "query": query
            }, indent=2)
        
        # Format results for agent
        formatted_results = []
        for product in results:
            formatted_results.append({
                "product_id": product.get("Product ID"),
                "name": product.get("Product Name"),
                "brand": product.get("Brand Name"),
                "price": f"{product.get('Price')}",
                "frame_type": product.get("Frame Type"),
                "frame_shape": product.get("Frame Shape"),
                "color": product.get("Frame Colour"),
                "rating": product.get("Rating"),
                "reviews": product.get("Number of Reviews"),
                "image_url": product.get("Image URL"),
                "match_score": f"{round(product.get('similarity_score', 0) * 100, 1)}%"
            })
        
        return json.dumps({
            "status": "success",
            "query": query,
            "results_count": len(formatted_results),
            "products": formatted_results
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Search failed: {str(e)}",
            "query": query
        }, indent=2)


@tool("visual_search_products", description="Find similar products based on image and optional text")
def visual_search_products(image_url: str, text_description: str = "") -> str:
    """
    Find visually similar products using CLIP embeddings.
    
    Args:
        image_url: URL of the reference product image
        text_description: Optional text like "but in blue color" or "cheaper alternatives"
    
    Examples:
    - visual_search_products("https://...", "similar but cheaper")
    - visual_search_products("https://...", "same style in brown")
    """
    print(f"🖼️  Visual search tool called")
    print(f"   Image: {image_url[:60]}...")
    print(f"   Text: {text_description or 'None'}")
    
    try:
        embedding_service = get_embedding_service()
        
        # Determine search mode
        if text_description:
            # Hybrid search: image + text
            results = embedding_service.search_by_image_and_text(
                image_url=image_url,
                text_query=text_description,
                top_k=5,
                text_weight=0.4  # 40% text, 60% image
            )
        else:
            # Pure image search
            results = embedding_service.search_by_image_and_text(
                image_url=image_url,
                top_k=5
            )
        
        if not results:
            return json.dumps({
                "status": "no_results",
                "message": "No similar products found.",
                "image_url": image_url
            }, indent=2)
        
        # Format results
        formatted_results = []
        for product in results:
            formatted_results.append({
                "product_id": product.get("Product ID"),
                "name": product.get("Product Name"),
                "brand": product.get("Brand Name"),
                "price": f"{product.get('Price')}",
                "frame_type": product.get("Frame Type"),
                "color": product.get("Frame Colour"),
                "rating": product.get("Rating"),
                "image_url": product.get("Image URL"),
                "similarity_score": f"{round(product.get('similarity_score', 0) * 100, 1)}%"
            })
        
        return json.dumps({
            "status": "success",
            "search_type": "visual" if not text_description else "hybrid",
            "results_count": len(formatted_results),
            "products": formatted_results
        }, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Visual search failed: {str(e)}",
            "image_url": image_url
        }, indent=2)