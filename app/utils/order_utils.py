"""
Utilities for working with order data
"""

from datetime import datetime
from typing import List, Dict, Optional


def calculate_days_since_delivery(delivery_date: str) -> Optional[int]:
    """
    Calculate number of days since delivery.
    
    Args:
        delivery_date: Date string in format YYYY-MM-DD
    
    Returns:
        Number of days, or None if not yet delivered
    """
    if not delivery_date:
        return None
    
    try:
        delivery = datetime.strptime(delivery_date, "%Y-%m-%d")
        today = datetime.now()
        delta = today - delivery
        return delta.days
    except Exception as e:
        print(f"Error parsing delivery date: {e}")
        return None


def check_return_eligibility(delivery_date: str, return_window_days: int = 10) -> Dict:
    """
    Check if an order is eligible for return based on delivery date.
    
    Args:
        delivery_date: Date string in format YYYY-MM-DD
        return_window_days: Number of days allowed for returns (default: 10)
    
    Returns:
        Dict with eligibility status and details
    """
    days_since = calculate_days_since_delivery(delivery_date)
    
    if days_since is None:
        return {
            "eligible": False,
            "reason": "Order not yet delivered",
            "days_since_delivery": None,
            "days_remaining": None
        }
    
    eligible = days_since <= return_window_days
    days_remaining = return_window_days - days_since
    
    return {
        "eligible": eligible,
        "reason": "Within return window" if eligible else "Return window expired",
        "days_since_delivery": days_since,
        "days_remaining": max(0, days_remaining),
        "return_window_days": return_window_days
    }


def check_replacement_eligibility(delivery_date: str, replacement_window_days: int = 15) -> Dict:
    """
    Check if an order is eligible for replacement based on delivery date.
    
    Args:
        delivery_date: Date string in format YYYY-MM-DD
        replacement_window_days: Number of days allowed for replacement (default: 15)
    
    Returns:
        Dict with eligibility status and details
    """
    days_since = calculate_days_since_delivery(delivery_date)
    
    if days_since is None:
        return {
            "eligible": True,  # Can replace damaged items even before delivery
            "reason": "Order not yet delivered - can claim damaged on arrival",
            "days_since_delivery": None,
            "days_remaining": None
        }
    
    eligible = days_since <= replacement_window_days
    days_remaining = replacement_window_days - days_since
    
    return {
        "eligible": eligible,
        "reason": "Within replacement window" if eligible else "Replacement window expired",
        "days_since_delivery": days_since,
        "days_remaining": max(0, days_remaining),
        "replacement_window_days": replacement_window_days
    }


def format_order_summary(order: Dict) -> str:
    """
    Format a single order into a human-readable summary.
    
    Args:
        order: Order dictionary
    
    Returns:
        Formatted string summary
    """
    summary = f"Order {order['order_id']}:\n"
    summary += f"  Product: {order['product_name']}\n"
    summary += f"  Status: {order['order_status']}\n"
    summary += f"  Ordered: {order['order_date']}\n"
    
    if order['delivery_date']:
        summary += f"  Delivered: {order['delivery_date']}"
        days = calculate_days_since_delivery(order['delivery_date'])
        if days is not None:
            summary += f" ({days} days ago)"
        summary += "\n"
    
    summary += f"  Amount: ₹{order['total_amount']}"
    if order['discount_applied'] and order['discount_applied'] > 0:
        summary += f" (Discount: ₹{order['discount_applied']})"
    summary += "\n"
    
    summary += f"  Payment: {order['payment_status']}\n"
    
    return summary


def format_orders_list(orders: List[Dict], max_orders: int = 5) -> str:
    """
    Format a list of orders into a numbered list.
    
    Args:
        orders: List of order dictionaries
        max_orders: Maximum orders to include (default: 5)
    
    Returns:
        Formatted string with numbered list
    """
    if not orders:
        return "No orders found."
    
    orders_to_show = orders[:max_orders]
    result = f"Your recent orders ({len(orders_to_show)} of {len(orders)}):\n\n"
    
    for idx, order in enumerate(orders_to_show, 1):
        result += f"{idx}. Order {order['order_id']} - {order['product_name']}\n"
        result += f"   Status: {order['order_status']}"
        
        if order['delivery_date']:
            days = calculate_days_since_delivery(order['delivery_date'])
            if days is not None:
                result += f" | Delivered {days} days ago"
        
        result += f" | ₹{order['total_amount']}\n\n"
    
    if len(orders) > max_orders:
        result += f"... and {len(orders) - max_orders} more orders\n"
    
    return result


def extract_product_preferences(orders: List[Dict]) -> Dict[str, any]:
    """
    Extract user preferences from order history.
    
    Args:
        orders: List of order dictionaries
    
    Returns:
        Dict with inferred preferences
    """
    if not orders:
        return {
            "has_orders": False,
            "total_orders": 0
        }
    
    # Count product types
    product_names = [order['product_name'].lower() for order in orders]
    
    # Simple categorization
    has_sunglasses = any('sunglass' in name for name in product_names)
    has_reading = any('reading' in name or 'reader' in name for name in product_names)
    has_computer = any('computer' in name or 'blue' in name for name in product_names)
    
    # Brand preferences
    brands = {}
    for order in orders:
        name = order['product_name']
        # Extract brand (usually first words)
        words = name.split()
        if len(words) >= 2:
            potential_brand = ' '.join(words[:2])
            brands[potential_brand] = brands.get(potential_brand, 0) + 1
    
    favorite_brand = max(brands.items(), key=lambda x: x[1])[0] if brands else None
    
    # Price range
    prices = [float(order['total_amount']) for order in orders if order['total_amount']]
    avg_price = sum(prices) / len(prices) if prices else 0
    
    return {
        "has_orders": True,
        "total_orders": len(orders),
        "has_sunglasses": has_sunglasses,
        "has_reading_glasses": has_reading,
        "has_computer_glasses": has_computer,
        "favorite_brand": favorite_brand,
        "average_price": round(avg_price, 2),
        "price_range": {
            "min": min(prices) if prices else 0,
            "max": max(prices) if prices else 0
        }
    }


def get_order_by_id(orders: List[Dict], order_id: str) -> Optional[Dict]:
    """
    Find a specific order by ID from the orders list.
    
    Args:
        orders: List of order dictionaries
        order_id: Order ID to find (case-insensitive)
    
    Returns:
        Order dict if found, None otherwise
    """
    order_id_upper = order_id.upper()
    for order in orders:
        if order['order_id'].upper() == order_id_upper:
            return order
    return None