from email import message
import operator
from re import search
from langchain_core.documents.base import BaseMedia
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatMessagePromptTemplate, ChatPromptTemplate
from typing_extensions import TypedDict, Annotated

from langgraph.graph.message import add_messages

from pydantic import BaseModel, Field

from typing import List

from dotenv import load_dotenv

import json
load_dotenv()

# --------------------
# Flow to follow
# --------------------

# --------------------
# 1) Create state: the data that moves between the agents (graph/state)
# --------------------


# --------------------
# 2) One agent at a time:
# schema.py: What the agent must return
# prompt.py: its instructions
# node.py: takes state, calls the LLM, checks the output, returns an update
# test it alone
# --------------------


# --------------------
# 3) Wiring: add the nodes and edges (builder.py)
# --------------------

# --------------------
# 4) build graph then ainvoke, astream
# --------------------

# --------------------
# 5) Debug: watch steps, then use langfuse
# --------------------

# Rule: state first, then agents one by one and wiring last

"""
What goes to the LLM

Role
System: Who the agent is and what to do

Human: This request's data: topic, facts findings

AI: The model replies

Tool: Results of tool calls


"""


# Create an Actual state
class ResearchState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages] # agent conversation will be saved
    topic: str # what to search
    search_queries: List[str]
    findings: Annotated[List[dict], operator.add] # finding should have reducer, When 2 search agents run in parrallel and both write to it. Which only accepts chat messages
    report: str
    analysis: str
    quality_score: float
    quality_feedback: str
    iteration: int


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

creative_llm = ChatOpenAI(model="gpt-40-mini", temperature=0.7)


SUPERVISOR_PROMPT = ChatPromptTemplate.from_messages([
        ("system", " You are research supervisor. Given a topic, generate 3 specific search queries that will cover different angles of the topic. Return ONLY JSON array of strings. No markdown formatting."),
        # keep a topic as a placeholder and not as f-string
        ("human", "Research topic: {topic}")
    ])


def supervisor(state: ResearchState) -> dict:
    """
    Plan the research: turn the topic into search queries.
    """


    # format messages takes keyword arguments, not dict
    messages = SUPERVISOR_PROMPT.format_messages(topic = state['topic'])
    response = llm.invoke(messages)

    queries = json.loads(response.content)

    return {
        "search_queries": queries[:3]
    }

def main():
    update = supervisor({"topic": "Solar vs wind energy in the GCC"})
    print(update)


if __name__ == "__main__":
    main()

