import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Optional
from langchain.tools import tool
from app.services.embedding_service import get_embedding_service
import csv

# Database paths
POLICIES_DATA = None
SAMPLE_DB_PATH = Path(__file__).parent.parent.parent / "sample_data.db"


def load_policies():
    """Load policies from CSV file OR from database table"""
    global POLICIES_DATA
    if POLICIES_DATA is not None:
        return POLICIES_DATA
    
    POLICIES_DATA = []
    
    # Try loading from database first
    try:
        conn = sqlite3.connect(str(SAMPLE_DB_PATH))
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute("SELECT * FROM policies")
        rows = cur.fetchall()
        for row in rows:
            POLICIES_DATA.append(dict(row))
        conn.close()
        
        if POLICIES_DATA:
            print(f"✅ Loaded {len(POLICIES_DATA)} policies from database")
            return POLICIES_DATA
    except Exception as e:
        print(f"⚠️  Could not load from database: {e}")
    
    # Fallback to CSV
    try:
        policies_path = Path(__file__).parent.parent / "data" / "policies.csv"
        with open(policies_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                POLICIES_DATA.append(row)
        print(f"✅ Loaded {len(POLICIES_DATA)} policies from CSV")
    except Exception as e:
        print(f"❌ Could not load policies: {e}")
    
    return POLICIES_DATA


def get_order_db_connection():
    """Get connection to sample_data.db"""
    conn = sqlite3.connect(str(SAMPLE_DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def fetch_user_orders(user_id: str, limit: int = 10) -> List[Dict]:
    """
    Fetch user's order history from orders_updated table.
    Joins with users table for customer info and shipping address.
    
    Returns list of orders with all details
    """
    try:
        conn = get_order_db_connection()
        cur = conn.cursor()
        
        # Get user info for customer name and shipping
        cur.execute("SELECT * FROM users WHERE user_id = ?", (user_id,))
        user = cur.fetchone()
        
        # Fetch orders from orders_updated table
        cur.execute("""
            SELECT 
                o.order_id, o.user_id, o.product_id, o.quantity,
                o.order_date, o.delivery_date, o.total_amount,
                o.discount_applied, o.payment_status, o.payment_type,
                o.created_at,
                p."Product Name" as product_name
            FROM orders_updated o
            LEFT JOIN updated_essilor_products p ON o.product_id = p."Product ID"
            WHERE o.user_id = ?
            ORDER BY o.created_at DESC
            LIMIT ?
        """, (user_id, limit))
        
        orders = []
        for row in cur.fetchall():
            # Determine order status based on delivery date
            order_status = "Delivered" if row.get('delivery_date') else "Processing"
            
            orders.append({
                'order_id': row['order_id'],
                'user_id': row['user_id'],
                'product_id': row['product_id'],
                'product_name': row.get('product_name', 'Unknown Product'),
                'customer_name': user['full_name'] if user else None,
                'email': user['email'] if user else None,
                'quantity': row['quantity'],
                'order_date': row['order_date'],
                'order_status': order_status,
                'delivery_date': row.get('delivery_date'),
                'total_amount': row['total_amount'],
                'discount_applied': row.get('discount_applied', 0),
                'payment_status': row.get('payment_status', 'Completed'),
                'payment_type': row.get('payment_type'),
                'shipping_address': {
                    'street': user['shipping_street'] if user else None,
                    'city': user['shipping_city'] if user else None,
                    'state': user['shipping_state'] if user else None,
                    'pincode': user['shipping_pincode'] if user else None,
                    'country': user.get('shipping_country', 'India') if user else 'India'
                },
                'created_at': row['created_at']
            })
        
        conn.close()
        return orders
    
    except Exception as e:
        print(f"❌ Error fetching orders: {e}")
        import traceback
        traceback.print_exc()
        return []


@tool("get_order_details", description="Get specific order details by order ID or track recent orders")
def get_order_details(query: str) -> str:
    """
    Get details about specific orders based on order ID or general tracking queries.
    
    Args:
        query: Order ID (e.g., "ORD001") or general query like "track my order", "order status"
    
    Examples:
    - "What's the status of order ORD001?"
    - "Track my order ORD123"
    - "Where is my order?"
    - "When will my order arrive?"
    
    Note: This tool expects user_id to be in the query context (set by the system)
    """
    print(f"📦 Order details tool called with query: {query}")
    
    # Extract order ID if present (pattern: ORD followed by digits)
    import re
    order_id_match = re.search(r'ORD\d+', query.upper())
    
    return json.dumps({
        "status": "success",
        "tool": "get_order_details",
        "query": query,
        "order_id": order_id_match.group(0) if order_id_match else None,
        "instruction": (
            "Use the user's order history from context to answer this question. "
            "If specific order ID is mentioned, focus on that order. "
            "Otherwise, show recent orders and their status. "
            "Be helpful with delivery dates, tracking info, and order status."
        )
    }, indent=2)


@tool("get_policy_info", description="Get warranty, return, replacement, and other policy information")
def get_policy_info(query: str) -> str:
    """
    Retrieve policy information about warranty, returns, replacements, etc.
    
    Args:
        query: User's policy question (e.g., "return policy for ORD001", "warranty", "can I return")
    
    Examples:
    - "What's the warranty period?"
    - "Can I return order ORD001?"
    - "I want to return my glasses from order ORD123"
    - "What if my glasses arrive damaged?"
    """
    print(f"📋 Policy tool called with query: {query}")
    
    # Extract order ID if present
    import re
    order_id_match = re.search(r'ORD\d+', query.upper())
    
    try:
        policies = load_policies()
        query_lower = query.lower()
        
        # Keyword matching for relevant policies
        relevant_policies = []
        
        policy_keywords = {
            "warranty": ["warranty", "defect", "manufacturing"],
            "return": ["return", "refund", "money back", "not satisfied"],
            "replacement": ["replacement", "replace", "damaged", "broken", "defective"],
            "custom": ["custom", "personalized", "engraved", "non-returnable"],
            "extended": ["extended", "additional coverage", "accidental damage"]
        }
        
        matched_types = set()
        for policy_type, keywords in policy_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                matched_types.add(policy_type)
        
        if not matched_types:
            relevant_policies = policies
        else:
            for policy in policies:
                policy_type_lower = policy["Policy Type"].lower()
                if any(ptype in policy_type_lower for ptype in matched_types):
                    relevant_policies.append(policy)
        
        if not relevant_policies:
            relevant_policies = policies
        
        response_data = {
            "status": "success",
            "tool": "get_policy_info",
            "query": query,
            "order_id": order_id_match.group(0) if order_id_match else None,
            "policies": relevant_policies,
            "instruction": (
                "Present these policies clearly to the user. "
                "Use bullet points for conditions. "
                "Highlight key timeframes (10 days, 15 days, 12 months). "
                "If order ID is mentioned, check the user's order history from context and provide specific guidance. "
                "For complex situations, offer to connect with customer support."
            )
        }
        
        # If order ID mentioned, add note about checking order details
        if order_id_match:
            response_data["note"] = (
                "Order ID detected. Use the user's order history from context "
                "to provide specific policy guidance for this order (e.g., delivery date for return window)."
            )
        
        return json.dumps(response_data, indent=2)
    
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to retrieve policy info: {str(e)}",
            "instruction": "Apologize and offer to connect user with customer support."
        }, indent=2)


@tool("handle_conversation", description="Handle conversational queries, use order history ONLY for shopping/order-related questions")
def handle_conversation(query_type: str, user_message: str) -> str:
    """
    Handle general conversation, greetings, and provide context-aware responses.
    
    Args:
        query_type: "greeting", "capability", "clarification", "follow_up", "history_reference"
        user_message: The original user message
    
    IMPORTANT: Order history is available but should be used selectively:
    - Pure greetings ("hi", "hello"): DO NOT mention order history
    - Shopping requests ("I need glasses"): Use order history to personalize
    - Order queries ("where's my order"): Use order history for details
    - Follow-ups about products: Use order history if relevant
    """
    print(f"💬 Conversation tool called")
    print(f"   Type: {query_type}")
    print(f"   Message: {user_message}")
    
    # Determine if this is a pure greeting with no shopping intent
    is_pure_greeting = query_type == "greeting" and user_message.lower().strip() in [
        'hi', 'hello', 'hey', 'hi there', 'hello there', 'good morning', 
        'good afternoon', 'good evening', 'greetings', 'howdy'
    ]
    
    instruction = "This is a conversational query. "
    
    if is_pure_greeting:
        instruction += (
            "DO NOT mention order history or past purchases. "
            "Keep the greeting brief, welcoming, and professional. "
            "List your capabilities and ask what they need help with."
        )
    elif query_type == "clarification":
        instruction += (
            "User has shopping intent. Use order history to personalize clarifying questions. "
            "Reference past purchases briefly to show you remember them, "
            "then ask what they're looking for today."
        )
    else:
        instruction += (
            "Use conversation history and order history appropriately. "
            "If discussing products, reference past purchases naturally. "
            "If just chatting, keep order history subtle."
        )
    
    return json.dumps({
        "status": "success",
        "tool": "handle_conversation",
        "query_type": query_type,
        "user_message": user_message,
        "is_pure_greeting": is_pure_greeting,
        "instruction": instruction
    }, indent=2)


@tool("search_products", description="Search and recommend products using semantic understanding and user preferences")
def search_products(query: str) -> str:
    """
    Search products using semantic embeddings with awareness of user's order history.
    
    Examples:
    - "Show me black wayfarer glasses"
    - "I need reading glasses for a round face"
    - "Lightweight sports sunglasses under 2000 rupees"
    
    Note: User's order history is available in context for better recommendations
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
            "products": formatted_results,
            "instruction": (
                "Present these products to the user. "
                "If user's order history shows similar preferences, mention that. "
                "Use order history to make personalized suggestions."
            )
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
    """
    print(f"🖼️  Visual search tool called")
    print(f"   Image: {image_url[:60]}...")
    print(f"   Text: {text_description or 'None'}")
    
    try:
        embedding_service = get_embedding_service()
        
        if text_description:
            results = embedding_service.search_by_image_and_text(
                image_url=image_url,
                text_query=text_description,
                top_k=5,
                text_weight=0.4
            )
        else:
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