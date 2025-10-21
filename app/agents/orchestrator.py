from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.tools import get_order_details, suggest_products


def create_orchestrator_agent():
    """Create and return the orchestrator agent with tools."""
    llm = ChatOpenAI(model="gpt-3.5-turbo")
    
    agent = create_agent(
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
            "6. DO NOT ALTER THE RESPONSE FROM THE TOOL IN ANY WAY\n\n"
            "EXAMPLES:\n"
            "User: 'Where is my order?'\n"
            "→ Call get_order_details('Where is my order?')\n"
            "→ Respond with the tool's result\n\n"
            "User: 'Show me laptops'\n"
            "→ Call suggest_products('Show me laptops')\n"
            "→ Respond with the tool's result"
        )
    )
    
    return agent