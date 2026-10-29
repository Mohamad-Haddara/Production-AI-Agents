from os import name
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, BaseMessage
from pydantic.v1 import DecimalIsNotFiniteError
from typing_extensions import TypedDict, Annotated
from typing import Literal
from pydantic import BaseModel, Field
import operator
import json
from dotenv import load_dotenv


load_dotenv()


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)



# ==========================================================================
# Pattern 1: Message Passing
# Agents communicate through a shared message list
# ==========================================================================

class MessagePassingState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    current_phase: str


def create_message_passing_pipeline():
    """Agents communicate by appending messags that others can read."""

    def researcher(state: MessagePassingState) -> dict:
        # Researches the topic and posts findings as messages

        response = llm.invoke([
            SystemMessage(content=(
                "you are a researcher. Read the user's question,"
                "research it, and post your findings. Keep it to 2-3 sentences."
            )),
            *state["messages"]
        ])


        return {
            "messages": [AIMessage(
                content=f"[RESEARCHER]: {response.content}",
                name = "researcher"
            )],
            "current_phase": "fact_checker"
        }
    
    def fact_checker(state:MessagePassingState) -> dict:
        """Reads the researcher's message and validate the claims"""
        response = llm.invoke([
            SystemMessage(content=(
                "You are a fact-checker. Read the researcher's message and validate the claims."
                "Keep it to 2-3 sentences."
            )),

            *state["messages"]
        ])

        return {
            "messages": [AIMessage(
                content = f"[FACT-CHECKER]: {response.content}",
                name = "fact_checker"
            )],
            "current_phase": "summarizer"
        }

    def summarizer(state:MessagePassingState) -> dict:
        """Read all the previous messages and create a final summary"""
        response = llm.invoke([
            SystemMessage(content= (
                "you are a summarizer. Read the researcher's findings and"
                "the fact-checker's review. Produce a final, accurate summary."
                "Keep it to 2-3 sentences."
            )),
            *state["messages"]
        ])


        return {
            "messages": [AIMessage(
                content = f"[SUMMARY]: {response.content}",
                name = "summaizer"
            )],
            "current_phase": "done"
        }


    
    graph = StateGraph(MessagePassingState)

    graph.add_node("researcher", researcher)
    graph.add_node("fact_checker", fact_checker)
    graph.add_node("summarizer", summarizer)

    graph.add_edge(START, "researcher")
    graph.add_edge("researcher", "fact_checker")
    graph.add_edge("fact_checker", "summarizer")
    graph.add_edge("summarizer", END)


    return graph.compile()



def message_passing_demo():
    """Demonstrates the message passing pipeline"""
    agent = create_message_passing_pipeline()

    print("Message Passing Demo:\n")

    result = agent.invoke({
        "messages": [HumanMessage(content="What are the main benefits of renewable energy?")],
        "current_phase": "researcher"
    })


    for msg in result["messages"]:
        if isinstance(msg, AIMessage):
            print(f"{msg.name}: {msg.content}\n")



# ==========================================================================
# Pattern 2: Shared State
# Agents communicate through a shared state dictionary (state fields)
# ==========================================================================
class SharedFieldsState(TypedDict):
    query: str
    # Each agent writes to it is own field - others (agents) can read it (this field)
    raw_data: Annotated[list[dict], operator.add]
    analysis: str
    recommendations: list[str]
    confidence_score: float

def create_shared_fields_pipeline():
    """Agents communicate through typed state fields, not messages"""
    
    # First node: Data Collector 
    def data_collector(state: SharedFieldsState) -> dict:
        """Collects data and writes to the raw_data field"""
        response = llm.invoke([
            SystemMessage(content=(
                "You are a data collector. Collect data from the user query and produce 3 data points"
                "as a JSON array of objects with 'source' and 'finding' keys."
                "Reurn ONLY the JSON array, no markdown."
            )
            ),
            HumanMessage(content=state["query"]),
        ])

        try:
            data = json.loads(response.content)
        
        except json.JSONDecodeError:
            data = [{"source": "llm", "finding": response.content}]

        # We give information to the next agent
        return {
            "raw_data": data
        }


    
    def data_analyst(state: SharedFieldsState) -> dict:
        """Reads raw data field, write analysis and confidence."""

        # It is going to read the raw data field and write the analysis and confidence
        data_summary = json.dumps(state["raw_data"], indent=2)

        response = llm.invoke([
            SystemMessage(content=(
                "you are data analyst. Analyze the collected data and provide:"
                "1) A brief analysis (2-3 sentences), and"
                "2) A confidence score from 0.0 to 1.0."
                "Format: ANALYSIS: <text>\nCONFIDENCE: <number>"
            )

            
            
            ),
            HumanMessage(content=f"Query: {state['query']}\n\nData:\n{data_summary}"),
        ])

        content = response.content
        analysis = content
        confidence = 0.7 # default

        if "CONFIDENCE" in content:
            parts = content.split("CONFIDENCE:")
            analysis = parts[0].replace("ANALYSIS", "").strip()
            try:
                confidence = float(parts[1].strip())
            except:
                confidence = 0.7

        
        return {"analysis": analysis, "confidence_score": confidence}

    
    def advisor(state: SharedFieldsState) -> dict:
        """Read analysis + confidence, writes recommendations."""
        response = llm.invoke([
            SystemMessage(content=(
                "You are strategic advisor. Based on the analysis and"
                "confidence score, provide 3 actionable recommendations."
                "Return them as a JSON array of strings."
                "Return ONLY the JSON array, no markdown"
            )),

            HumanMessage(
                content=
                (
                    f"Query: {state['query']}\n"
                    f"Analysis: {state['analysis']}"
                    f"Confidence: {state['confidence_score']}"
                )
            )
        ])

        try: 
            recs = json.loads(response.content)
        except json.JSONDecodeError:
            recs = [response.content]


        return {"recommendations": recs}
        


    # Create a graph
    graph = StateGraph(SharedFieldsState)

    graph.add_node("data_collector", data_collector)
    graph.add_node("analyst",data_analyst)
    graph.add_node("advisor", advisor)

    graph.add_edge(START, "data_collector")
    graph.add_edge("data_collector", "analyst")
    graph.add_edge("analyst", "advisor")
    graph.add_edge("advisor", END)

    return graph.compile()    


def demo_shared_state():
    # Demo shared state fields between agents
    
    agent = create_shared_fields_pipeline()

    print("Shared State Demo:\n")

    result = agent.invoke({
        "query":"Should a small business invest in AI Automation in 2026?",
        "raw_data": [],
        "analysis": "",
        "recommendations": [],
        "confidence_score": 0.0
    })


    print(f"Data collected: {len(result['raw_data'])} points")
    for d in result['raw_data']:
        print(f" -[{d.get("source", "N/A")}] {d.get("finding", "N/A")[:80]}...")

    print(f"\nAnalysis: {result['analysis'][:200]}...")
    print(f"Confidence: {result['confidence_score']}")


    print("Recommendations:")
    for i, rec in enumerate(result['recommendations'], 1):
        print(f"    {i}: {rec}")



# =====================================================================
# Pattern 3: Blackboard Pattern
# Combines shared workspace + messages + iterative refinement
# =====================================================================

# operator.add it will keep every draft, and we want to keep the history of revision
# For messages ~ add_message --> we don't want to keep every draft, we want to make sure that the system itself is going to be able to update as we go.
class BlackboardState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # Reducer - conversation log
    # Blackboard fields - The shared workspce
    topic: str # What to write about
    drafts: Annotated[list[str], operator.add] # draft is accumulated
    critiques: Annotated[list[str], operator.add] # Critic is accumilated
    iteration: int # Counter - overwrite - we only care about current number
    is_approved: bool # gate flag


def create_blackboard_system():
    """
    Blackboard pattern: Multiple agents read/write a shared workspace.
    A drafter writes, a critic reviews, and the iterate until approved.

    """
    # Allow us to create a structured output
    class ApprovalDecision(BaseModel):
        approved: bool = Field(description="Whether the draft is good enough")
        feedback: str = Field(description = "Specific feedback if not approved")

    critic_llm = llm.with_structured_output(ApprovalDecision)

    def drafter(state: BlackboardState) -> dict:
        """Reads critiques from blackboard, writes improved draft."""
        context_parts = [f"Topic: {state['topic']}"]

        if state['drafts']:
            context_parts.append(f"Previous draft: {state['drafts'][-1]}")

        if state['critiques']:
            context_parts.append(f"Feedback to address: {state['critiques'][-1]}")

        
        context = "\n".join(context_parts)

        # Pass all that through human message
        response = llm.invoke([
            SystemMessage(content=(
                "You are a skilled writer. Write or revise a short paragraph"
                "(3-4 sentences) based on the topic and any feedback provided."
                "If there's feedback, directly address it in your revision."
            )),
            HumanMessage(content = context),
        ])



        return {
            "drafts": [response.content],
            # messages fields will have a list of AI messages
            "messages": [AIMessage(
                content = f"[DRAFTER iteration {state['iteration'] + 1}]: {response.content}",
                name = "drafter"
            )],
            "iteration": state['iteration'] + 1 # increment iteration state by 1
        }



    
    def critic(state: BlackboardState) -> dict:
        """Reads latest draft from blackboard, writes critique or approves."""
        last_draft = state['drafts'][-1] if state['drafts'] else "No draft yet"

        decision = critic_llm.invoke([
            SystemMessage(content=(
                "You are a strict editor. Review the draft for clarity, accuracy,"
                "and engagement. Approve ONLY if it is genuinely good."
                "If iteration is 3 or more, be more lenient."
                )),

            HumanMessage(content=(
                f"Topic: {state['topic']}\n"
                f"Iteration: {state['iteration']}\n"
                f"Draft: {last_draft}" # we get from draft state, the last one
            ))
        ])


        # Force approval after 3 iterations to prevent infinite loops (V.I)
        approved = decision.approved or state['iteration'] >=3

        result = {
            "is_approved": approved,
            "messages": [AIMessage(
                content=f"[CRITIC]: {'Approved' if approved else "REVISION NEEDED"}",
                name = "critic" 
            )]
        }

        if not approved:
            result['critiques'] = [decision.feedback]

        return result

    
    def route_after_critic(state: BlackboardState) -> Literal["drafter", "end"]:
        """Loop back to drafter if not approved."""
        if state["is_approved"]:
            return "end"
        
        return "drafter"

    
    graph = StateGraph(BlackboardState)

    graph.add_node("drafter", drafter)
    graph.add_node("critic", critic)
    
    graph.add_edge(START, "drafter")
    graph.add_edge("drafter", "critic")
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {"drafter":"drafter", "end": END}
    )

    
    return graph.compile()


def blackboard():
    agent = create_blackboard_system()

    print("Blackboard Pattern Demo:\n")

    result = agent.invoke({
        "messages": [],
        "topic":"Why LangGraph is great for building multi-agent systems",
        "drafts": [],
        "critiques":[],
        "iteration": 0,
        "is_approved": False
    })

    print(f"total iterations: {result['iteration']}")
    print(f"Approved: {result['is_approved']}")

    print("\nConversation:")
    for msg in result['messages']:
        if isinstance(msg, AIMessage):
            print(f"\n{msg.content}")

    print(f"\nFinal draft:\n{result['drafts'][-1]}")

"""
How it works?
- First we have the draft iteration -> drafter will read critiques from blackboard or dashboard, and writes improved draft.
- Critique will read it and say it needs revision and tell the drafter that it requires improvement
- Drafter will do second iteration - rewrite what critic said
- Critic will read it then approve it 
- Then publish final draft

Like we talked about shared workspace, one of the 3 pillar protocols when it comes to commuincation between agents. This is what we call an iterative refinement loop,
because two agents - we have a drafter and critic - they are passing work back and forth on a shared workspace until the quality is met. It is like a writer and editor
work together, they go back and forth.

START -> Drafter -> then passed to critic -> is it good enough? -> No -> Go back to the drafter -> If it is good -> END

This pattern is blackboard which is little different because now we have two agents that iterate through each other in a loop, until a quality gate is passed

Blackboard - where both agents read from it and write to it. Drafter read the last item in critics list to know what to fix and critic is going to read the last draft, so it knows what to review

Neither agent talk to other, they communicate entirely through the workspace
"""


if __name__ == "__main__":
    #message_passing_demo()
    #demo_shared_state()
    blackboard()