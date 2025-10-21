import getpass
import os

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

# from langchain_openai import ChatOpenAI
# from pydantic import BaseModel


# def get_weather(location: str) -> None:
#     """Get weather at a location."""
#     return "It's sunny."


# class OutputSchema(BaseModel):
#     """Schema for response."""

#     answer: str
#     justification: str


# llm = ChatOpenAI(model="gpt-4.1")

# structured_llm = llm.bind_tools(
#     [get_weather],
#     response_format=OutputSchema,
#     strict=True,
# )

# # Response contains tool calls:
# tool_call_response = structured_llm.invoke("What is the weather in SF?")

# # structured_response.additional_kwargs["parsed"] contains parsed output
# structured_response = structured_llm.invoke(
#     "What weighs more, a pound of feathers or a pound of gold?"
# )
# print(tool_call_response)

# from langchain_openai import ChatOpenAI, custom_tool
# from langchain.agents import create_agent


# @custom_tool
# def execute_code(code: str) -> str:
#     """Execute python code."""
#     return "27"


# llm = ChatOpenAI(model="gpt-5", use_responses_api=True)

# agent = create_agent(llm, [execute_code])

# input_message = {"role": "user", "content": "Use the tool to calculate 3^3."}
# for step in agent.stream(
#     {"messages": [input_message]},
#     stream_mode="values",
# ):
#     step["messages"][-1].pretty_print()


from langchain.tools import tool
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI



# Create a small dedicated sub-agent LLM. Using a smaller model name here
# keeps cost down; change to your preferred model in production.
subagent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
main_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Build a lightweight subagent (no external tools) that will be invoked by the
# main agent when the `call_subagent1` tool is used.
subagent1 = create_agent(subagent_llm, tools=[])


@tool(
    "subagent1_name",
    description="Delegate queries to the subagent1 conversational agent",
)
def call_subagent1(query: str) -> str:
    """Call the subagent with a user query and return the final message text."""
    # The return shape from `invoke` can vary by langchain version. This code
    # tries to handle the common dict-with-messages shape. If your version
    # returns an AgentResult/Response object adjust accordingly.
    result = subagent1.invoke({"messages": [{"role": "user", "content": query}]})
    # If `result` is a mapping with a 'messages' list:
    if isinstance(result, dict) and "messages" in result and result["messages"]:
        return getattr(result["messages"][-1], "content", result["messages"][-1])
    # Fallback: string-like result
    return str(result)


# Create the main agent and register the tool that delegates to subagent1.
agent = create_agent(main_llm, tools=[call_subagent1])


if __name__ == "__main__":
    # Simple interactive demo to exercise the agent (safe no-network fallback
    # depends on having OPENAI_API_KEY set and langchain installed).
    try:
        query = input("Ask the main agent (or type 'quit'): ")
        while query and query.lower() != "quit":
            out = agent.invoke({"messages": [{"role": "user", "content": query}]})
            print("Agent response:\n", out)
            query = input("Ask the main agent (or type 'quit'): ")
    except Exception as e:
        print("Error running agent demo:", e)