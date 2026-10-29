"""
Tool-Calling Agents with LangGraph
Building agents that can use tools
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages # 
from typing import Literal
import operator
import json
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)


# Define tool = function + tool decorator
@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Example: calculate ('2+2')"""
    try:
        result = eval(expression) # Note: In production, use a safe math parser
        return f"The result of {expression} is {result}"

    except Exception as e:
        return f"Error calculating: {e}"



# Define tool 2
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""

    # Simulated weather data
    weather_data = {
        "new_york": "72F, Sunny",
        "london": "58F, Cloudy",
        "tokyo": "68F, Clear",
        "paris": "65F, Rain"
    }


    city_lower = city.lower()
    if city_lower in weather_data:
        return f"Weather in {city}: {weather_data[city_lower]}"
    
    return f"Weather data not available for {city}"


# Search web tool
def search_web(query: str) -> str:
    """Simulate a web search for a query."""
    # simulated result
    search_result = {
        "Python programming": "xyz",
        "latest news": "xxx-yyy-zzzzzz",
        "best restaurant in new york": "tttt eeee wwww qqqq",
    }

    query_lower = query.lower()

    if query_lower in search_result:
        return f"Search result for: {query}: {search_result[query_lower]}"
    
    return f"No search results found for {query}"



class AgentState(TypedDict):
    #messages: Annotated[list[BaseMessage], operator.add] # reduce operator - append all messages coming in
    messages: Annotated[list[BaseMessage], add_messages] # use add_message -> Merges two lists of messages, updating existing messages by ID.


def create_tool_agent():
    # Create a basic tool-calling agent.

    # add all tools as a list
    tools = [calculate, get_weather, search_web]

    llm_with_tools = llm.bind_tools(tools) # This is the secret!


    # Define the actual nodes
    def agent_node(state: AgentState) -> dict:
        # Generate a response using the LLM with tool access
        response = llm_with_tools.invoke(state["messages"])
        #return {"response": response}
        return {"messages": [response]}
    
    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        """Check if we should continue to tools or end."""
        last_messages = state["messages"][-1]

        # If no tool calls. we are done
        if not hasattr(last_messages, "tool_calls") or not last_messages.tool_calls:
            return "end"
        return "tools"

    # Create tool node
    tool_node = ToolNode(tools)


    # Create a graph
    graph = StateGraph(AgentState)

    # Add nodes and edges
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)


    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END 
        }
    )

    graph.add_edge("tools", "agent") # loop back after tool execution

    return graph.compile()


def tool_agent():
    """Demo the tool-calling agent."""

    agent = create_tool_agent() # return graph.compile


    queries = [
        "What is 15*20?",
        "what is the weather in tokyo?",
    ]

    print("Tool-Calling Agent:\n")

    for query in queries:
        print(f"Query: {query}")

        result = agent.invoke({
            "messages": [HumanMessage(content=query)]
        })


        # Get Final Response
        final_message = result["messages"][-1]
        print(f"Response: {final_message.content}")
        print(f"Total messages: {len(result['messages'])}")
        print("-"*40)


# Execution Trace
def tool_execution_trace():
    """Showed detailed tool execution trace."""

    agent = create_tool_agent()

    print("\nTool Execution Trace:\n")

    result = agent.invoke(
        {
            "messages": [HumanMessage(content = "Calculate 15% of 250 and check weather in Paris")]
        }
    )

    # here where everything happens - giving us an index and message object - bcz everything loaded in a state object in form of messages (which use reducer)
    for i, msg in enumerate(result['messages']):
        msg_type = type(msg).__name__
        print(f"\n[{i}] : {msg_type}:")

        if isinstance(msg, HumanMessage):
            print(f"    Content: {msg.content}")

        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                print(f"    Tool calls: {len(msg.tool_calls)}")
                for tc in msg.tool_calls:
                    print(f"    -{tc['name']}({tc['args']})")
            
            else:
                print(f"    Content: {msg.content}")

        elif isinstance(msg, ToolMessage):
            print(f"    Tool: {msg.name}")
            print(f"    Result: {msg.content}")

# Reason - to pick a tool
# Act - to execute those tools
# Reason again to create an actual answer 
# Input -> tool call -> result -> final answer 







# Custom Tool with Error Handling
@tool
def divide(a: float, b: float) -> str:
    """Divide two numbers."""
    # error handling if the number is not divided by 0
    if b == 0:
        return "Error: Division by zero"
    
    result = a/b
    return f"The result of {a} divided by {b} is result {result}"


def tool_with_error():
    """Tool error handling"""

    tools = [divide]
    llm_with_tools = llm.bind_tools(tools)


    # Create an Agent Node
    def agent_node(state: AgentState) -> dict:
        
        response = llm_with_tools.invoke(state['messages'])
        return {"messages": [response]} # return list of responses


    def should_continue(state: AgentState) -> Literal["tools", "end"]:
        last_messages = state["messages"][-1]

        if not hasattr(last_messages, "tool_calls") or not last_messages.tool_calls:
            return "end"
        return "tools"

    tool_node = ToolNode(tools)


    graph = StateGraph(AgentState)
    
    graph.add_node("agents", agent_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "agents")
    graph.add_conditional_edges(
        "agents",
        should_continue,
        {
            
            "tools": "tools",
            "end": END,
        }
    )

    graph.add_edge("tools", "agents")


    agent = graph.compile()

    print("\nTool Error Handling:\n")

    queries = [
        "Divide 100 by 5",
        "Divivde 100 by 0" # will trigger error
    ]


    for query in queries:
        result = agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"Query: {query}")
        print(f"Response: {result['messages'][-1].content}")
        print("-"*40)
    """
    Agent doesn't crash, It doesn't need try/except on graph itself.
    The tool handle error interanlly and return string instead of crash. And that information is returned to llm and explain the error in natural language.

    V.I:
    1) Tools should return errors, not raise them. Because if tool raise a valueerror and then can divide by 0 -> so the entire graph would crash. By returning error string
    I keep agent alive
    
    2) LLM is resilient ~ meanining it reads the tool result - Recognize error message - then send response.

    So no need to write error handling logic and graph

    This is production pattern - real world tools - API calls - database queries - files reads

    Rule is : Don't just return result, but make sure that I return string with some explanation. So handle error in tool so the LLM can reason about them instead of whole system crashing.
    """


if __name__ == "__main__":
    #tool_agent()
    #tool_execution_trace()
    tool_with_error()