"""
Human-in-the-Loop Patterns in LangGraph
Interrupt, review, modify, and resume
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver # For saving intermediate state of the workflow of each node
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from typing import Literal
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model="gpt-4o-mini", temperature = 0.0)


class ApprovalState(TypedDict):
    request: str # what he user asked for
    draft: str  # LLM generate it
    approved: bool # Human yes or no decision
    feedback: str   # human revision notes
    final: str      # finished result



def interrupt_for_approval():
    # Interrupt execution for human approval

    # Create Node: Create draft
    def create_draft(state:ApprovalState) -> dict:
        response = llm.invoke(
            f"Create a professional response for: {state['request']}"
        )

        return {"draft": response.content}

    def wait_for_approval(state: ApprovalState) -> dict:
        # This node is where we will interupt - pause point
        return state


    def finalize(state:ApprovalState) -> dict:
        if state['approved']:
            return {"final":state['draft']}
        
        else:
            # In corporate feedback
            response = llm.invoke(
                f"Revise this draft based on feedback:\n\n"
                f"Draft: {state['draft']}\n\n"
                f"Feedback: {state['feedback']}"
            )

            return {'final': response.content}

    
    graph = StateGraph(ApprovalState)


    graph.add_node("draft", create_draft)
    graph.add_node("approval", wait_for_approval)
    graph.add_node("finalize", finalize)


    graph.add_edge(START, "draft")
    graph.add_edge("draft", "approval")
    graph.add_edge("approval","finalize")
    graph.add_edge("finalize", END)


    # Check pointer
    memory = MemorySaver() # To save intermediate states for reviews

    app = graph.compile(
        checkpointer = memory,  # set the MemorySaver
        interrupt_before = ["approval"],    # Set the interupt ~ when to interupt happen - Interrupt/pause before this node
    )


    print("Human-in-the-Loop Approval:\n")

    # Configuration for this thread - in production I would have a function that creates a unique thread for each time
    config = {"configuration": {"thread_id":"demo-1"}}

    # Step 1: run until interrupt
    print("Step 1: Running until approval checkpoint...")
    result = app.invoke(
        {
            "request":"Write a thank-you email for a job interview",
            "draft": "",
            "approved":False,
            "feedback": "",
            "final": ""
        },
        config
    )


    print(f"Draft created:\n {result['draft'][:200]}...")
    print("\n[Execution paused for human review]")


    # VERY IMPORTANT
    # Step 2: Get current state ~ knowing which thread_id to get state from
    current_state = app.get_state(config)
    print(f"\nCurrent node: {current_state.next}")


    # step 3: Simulate human feedback and continue
    print(f"\nStep 2: Human provides feedback and continues...")

    # Update state with human input ~ place where human do approval + feedback - in production I should extract from human input to update this state
    app.update_state(
        config,
        {
            "approved": False, # Request changes
            "feedback": "Make it more concise and add specific mention of the company culture"
        }
    )

    # Continue execution ~ None mean do not add any new input just continue with execution
    final_result = llm.invoke(None, config)

    print(f"\nFinal result:\n {final_result['final']}")



if __name__ == "__main__":
    interrupt_for_approval()
