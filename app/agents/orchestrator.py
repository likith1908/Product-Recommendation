import yaml
from pathlib import Path
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from app.agents.tools import (
    get_order_details, 
    search_products, 
    visual_search_products, 
    handle_conversation,
    get_policy_info  # ← New import
)

def load_prompt_config(name: str):
    prompt_path = Path(__file__).parent.parent / "prompts" / f"{name}.yaml"
    with open(prompt_path) as f:
        return yaml.safe_load(f)

def create_orchestrator_agent():
    """Create and return the orchestrator agent with all tools."""
    config = load_prompt_config("orchestrator_prompt_v2")

    llm = ChatOpenAI(model=config["model"], temperature=config["temperature"])
    
    agent = create_agent(
        model=llm,
        tools=[
            handle_conversation,
            search_products, 
            visual_search_products, 
            get_order_details,
            get_policy_info  # ← Add policy tool
        ],
        system_prompt=(
            config["system_prompt"]
        )
    )
    
    return agent