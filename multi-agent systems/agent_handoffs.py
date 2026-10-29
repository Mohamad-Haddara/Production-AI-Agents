"""
Agent handoffs in LangGraph
Passing control and context between agents
"""

from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# SHARED MESSAGE LIST
class HandoffState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # appending conversation history
    current_agent: str  # handling the issue right now
    handoff_reason: str # reason why is transferred to the next agent
    context_summary: str # key info for the next agent
    # in production I can have:
    #priority: int
    # customer_id: int
    # more fields

# Triage agent needs to make a decision ~ we can't trust text for routing
# Thats why I use structured output 
class HandoffDecision(BaseModel):
    handoff_to: Literal["sales", "support", "billing", "stay", "end"] = Field(
        description="Which agent to hand off to"
    )

    reason: str = Field(description="Reason for handoff")
    context: str = Field(description="Key context to pass to next agent")


def create_customer_service_system():

    # Create actual agent or node - Bain of the system
    def triage_agent(state: HandoffState) -> dict:
        """Initial triage to route customer."""

        # system prompt to understand customer need
        # Extremely Important - Clearer description = better routing accuracy
        system = """
            You are a customer service triage agent. Your job is to understand:
            1. Understand the customer's need 
            2. Route to the appropriate specialist:
                - sales: Product questions, purchases, upgrades
                - support: Technical issues, bugs, how-to questions
                - billing: Payments, invoices, refunds
                - end: Simple questions you can answer directly

            Analyze the customer's message and decide where to route them.
        """
        # result will follow the structured I defined 
        handoff_llm = llm.with_structured_output(HandoffState)

        messages = [SystemMessage(content=system)] + state["messages"]
        decision = handoff_llm.invoke(messages)

        if decision == "end":
            response = llm.invoke([
                SystemMessage(content="Provide a brief, helpful response to the customer.")
                *state["messages"]
            ])

            return {
                "messages": [AIMessage(content=f"[Triage] {response.content}")],
                "current_agent": "end"
            }

        
        return {
            "current_agent": decision.handsoff_to,
            "handoff_reason": decision.reason,
            "context_summary": decision.context,
            "messages": [AIMessage(content=f"[Triage]   Transferring to {decision.handsoff_to}")]

        }

    def sales_agent(state: HandoffState) -> dict:
        """Sales specialist"""
        system_prompt = f"""
            You are sales specialist. Context from triage: {state.get("context_summary")}.

            Help the customer with product questions and purchases.
            Be helpful and informative, not pushy.

        """

        response = llm.invoke([
            SystemMessage(content=system_prompt),
            *state['messages']
        ])

        return {
            "messages": [AIMessage(content= f"[Sales] {response.content}")],
            "current_agent": "sales_complete"
        }


    def support_agent(sate:)