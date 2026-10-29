from dotenv import load_dotenv
from langgraph.graph import MessagesState # is an object has dictionary of key messages, and messages will be list of base messages - track on the state on all messages between agent(HumanMessage and AIMessage)
from langgraph.prebuilt import ToolNode # Execute tool - it is going to check the last message between the agent and human, and if the last message which is AIMessage that has a valid tool call, it is going to execute that tool call. If it decide to run tavily, it will run inside ToolNode

from lang_graph.langgraph_core import MessageState
from react import llm, tools # import llm and list of tools defined 

load_dotenv()


# Implement Reasoning Engine

# Node ~ where an agent gonna recieve a message, and it is going to decide whether it can answer it or it invoke a tool

SYSTEM_MESSAGE = """
You are helpful assistant that can use the following tools to answer questions.
"""


# Define a first node - with MessageState ~ which is a dictionary that has the key of messages
def run_agent_reasoning(state: MessageState) -> MessageState:
    """
    Run the agent reasoning node.
    """
    response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, * state['messages']]) # we put system messages and all the messages that we have
    # First iteration we will only have the human message
    # Second iteration we will have human and AI messages (AI response)
    # state["messages"] -> this is going to send the human message back to the LLM, then LLM decide whether we need to use a function call and execute a function or it decides that can output an answer


    # After node run, we need to update our state - if we return a dictionary with the key of message and return here a list
    # This list will have a response from llm
    return {
        "messages": [response] # AIMessage, and langGraph will append this value to our state in the messages key. So the key will have all messages and this node is going to update it and add another message

    }

# Execute tool that we need
tool_node = ToolNode(tools) # object class from langGraph prebuilt and provided with tool we defined (langchain tools)
# it can run tools in parallel, can run them in streaming mode