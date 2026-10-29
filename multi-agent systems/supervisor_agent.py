"""
Supervisor Architecture in LangGraph
One agent coordinate multiple specialist agents.

One Supervisor agent and multiple agent specialist below to report back to the supervisor
"""

from email import message
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from openai.types import ChatModel
from openai.types.shared import reasoning
from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages
from typing import Literal
from pydantic import BaseModel, Field
import operator
from dotenv import load_dotenv


load_dotenv()

#1) Actual state - is very important for our workflow bcz every agent write to and read from the same state - That's how communicate, not by talking directly to each other, BUT THROUGH THE SHARED MESSAGE LIST
class SupervisorState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # merge list of messages
    next_agent: str # what is the next agent need to do a certain task
    task_complete: bool
    final_response: str


def create_supervisor_system():
    """Create a supervisor with specialist agents."""

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Defining routing schema
    class RouteDecision(BaseModel):
        next: Literal["researcher", "writer", "critic", "FINISH"] = Field(
            description="The next agent to call, or FINISH if task is complete"
        )

        reasoning: str = Field(description="Why this agent was chosen")


    # Create a structure output which is going to be driven by RouteDecision  - Supervisor return a structured output and not the free text (we make route decision deterministic)
    supervisor_llm = llm.with_structured_output(RouteDecision)

    # 2) Supervisor node - Actual supervisor - is the Brain that never does the work (doesn't do research, ...), it just decide who goes next and it uses structured output to gurantee to return one of (researcher, writer, critic, finish)
    def supervisor(state: SupervisorState) -> dict:

        system_prompt = """
        You are a supervisor managing a team specialists:

        1. Researcher - Gathers information and facts
        2. Writer - Creates content and text
        3. critic - Reviews and improves work

        Based on the conversation, decide which agent should act next.
        If the task is complete, respond with FINISH.

        Current conversation shows the progress so far.

        """

        messages = [SystemMessage(content=system_prompt)] + state["messages"] # append any messages from the state

        decision = supervisor_llm.invoke(messages)

        if decision == "FINISH":
            return {
                "next_agent": "FINISH",
                "task_complete": True
            }

        
        return {
            "next_agent": decision.next,
            "messages": [AIMessage(content=f"[Supervisor] Routing to {decision.next}: {decision.reasoning}")] # we put this in our messages state
        }

    # Define the specialist agents ~ Demo purposes
    def researcher(state: SupervisorState) -> dict:
        # prompt with system and human message to the LLM
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are research specialist. Gather facts and information relevant "),
                ("human", "Task content:\n{context}\n\nProvide our research findings."),
            ]
        )

        # Get the task from first human message
        task = next((m.content for m in state['messages'] if isinstance(m, HumanMessage)))

        response = llm.invoke(prompt.format_messages(context = task))

        return {
            "messages": [AIMessage(content=f"[Researcher] {response.content}")] # append researcher to the messages
        }

    def writer(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are writing specialist. Create a clear engaging content "),
                ("human", "Previous work:\n{context}\n\nWrite the content"),
            ]
        )


        context = "\n".join([m.content for m in state["messages"][-5:]])
        response = llm.invoke(prompt.format_messages(context= context))


        return {
            "messages": [AIMessage(content=f"[Writer] {response.content}")]
        }

    

    def critic(state: SupervisorState) -> dict:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "You are a quality critic. Review the work and provide constructive feedback."),
                ("human", "Work to review:\n{context}\n\nProvide your critique."),
            ]
        )

        
        context = "\n".join([m.content for m in state['messages'][-3:]])
        response = llm.invoke(prompt.format_messages(context= context))

        return {
            "messages": [AIMessage(content=f"[Critic] {response.content}")]
        }

    
    def finalize(state: SupervisorState) -> dict:
        # Get the last response
        for msg in reversed(state["messages"]):
            if isinstance(msg, AIMessage) and "Writer" in msg.content:
                content = msg.content.replace("[Writer] ", "")
                return {"final_response": content}

        return {"final_response": "Task completed."}

    
    # route to agent based on supervisor decision
    def route_to_agent(state: SupervisorState) -> str:
        if state.get("task_complete"):
            return "finalize"
        
        return state["next_agent"]

    
    # Build graph
    graph = StateGraph(SupervisorState)
    graph.add_node("supervisor", supervisor)
    graph.add_node("researcher", researcher)
    graph.add_node("writer", writer)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "researcher":"researcher",
            "writer": "writer",
            "critic": "critic",
            "FINISH": "finalize"
        }
    )

    # Pattern
    #3) After each specialist I need to go to supervisor (no specialist can call another specialist directly)
    graph.add_edge("researcher","supervisor")
    graph.add_edge("writer","supervisor")
    graph.add_edge("critic","supervisor")
    graph.add_edge("finalize",END)


    return graph.compile()



def supervisor():
    # Supervisor system
    agent = create_supervisor_system()

    print("Supervisor Agent: \n")

    result = agent.invoke(
        {
            "messages": [HumanMessage(content="Write a short blog post about the benefits of AI in healthcare")],
            "next_agent": "",
            "task_complete":False,
            "final_response": ""
        }
    )



    print("Agent conversation:")
    for msg in result['messages']:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content[:200]}...")

    
    print(f"\n\nFinal Response:\n{result['final_response']}")


def supervisor_trace():
    # Show supervisor decision-making

    agent = create_supervisor_system()

    print("\nSupervisor Decision Trace:\n")


    result = agent.invoke({
        "messages": [HumanMessage(content="Create a marketing tagline for a new coffee branch")],
        "next_agent": "",
        "task_complete":False,
        "final_response": ""
    })


    print("Routing decisions:")

    for msg in result['messages']:
        if isinstance(msg, AIMessage) and "[Supervisor]" in msg.content:
            print(f"    -> {msg.content}")


"""
Architecture used behind
3 reasons of this pattern:
1) Separation of concerns - Each agent has one job and a focus prompt. (smaller scope better output)
2) Gain dynamic workflows - supervisor decide the flow at runtime
3) All production and Enterprise use this pattern ~ one orchestrator and multiple specialists
"""
if __name__ == "__main__":
    #supervisor()
    supervisor_trace