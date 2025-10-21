from langchain.tools import tool


@tool("order_details", description="Get details about a specific order")
def get_order_details(query: str) -> str:
    """Get details about a specific order based on the user query."""
    print("I am the order tool, I got a request for:", query)
    return f"[ORDER_TOOL] Order details for: {query}"


@tool("product_suggestions", description="Suggest products based on user query")
def suggest_products(query: str) -> str:
    """Suggest products to the user based on their query."""
    print("I am the product tool, I got a request for:", query)
    return f"[PRODUCT_TOOL] Product suggestions for: {query}"