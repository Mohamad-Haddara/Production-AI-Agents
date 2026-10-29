# Agent Brain: Implementing the ReAct Runnable
# We are going to use function calling that is going to be reasoning engine which decides which tool to call

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

load_dotenv()
# the function will be bind to llm so llm deicde whether to use the function or not

# turn it into langchain tool
@tool
def triple(num:float) -> float:
    """
    param num: a number to triple
    returns: the triple of the input number

    """

    return float(num) * 3


# Tavily search tool - limiting one fetch result
tools = [TavilySearch(max_result=1), triple]


# tell llm that has access to these tools
# We are going to use function calling

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools) # LLM handle the parsing
