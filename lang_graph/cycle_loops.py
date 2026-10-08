"""
Until now was a straight line. We have input that goes in and flows through nodes, comes out.

Now we are going to talk about 'cycles' and 'loops'.

Cycle and Loops where langGraph becomes very powerful.
Idea is the agent can go back and try again until it gets it right.

That what separate simple chain into actual intelligent agent.


Cycles and Loops in LangGraph
Self-correcting agents and iterative refinement
"""


from typing import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph,START, END
from typing_extensions import TypedDict, Annotated
from typing import Literal 
from langchain.messages import HumanMessage, AIMessage, SystemMessage
import operator
from dotenv import load_dotenv

load_dotenv()




llm = init_chat_model(model="gpt-4o-mini", temperature = 0.0)


# Build self-correcting code generator - So we have agent writes code - Check if its valid & loops back to fix its own mistakes
# Self-correcting agent pattern
# One of the most valuable real world patterns


# Step 1: Create a State
class CodeGenState(TypedDict):
    task: str
    code: str
    errors: Annotated[list[str], operator.add] # list concatenation of those errors - using reducer ~ operator.add ~ means every error gets appended and not repalced ~ real power. This give agent memory of all past failure
    iteration: int
    max_iteration: int # without this -> the agent will loop forever
    success: bool


def self_correcting_code():
    # Self-correcting code generator


    def generate_code(state: CodeGenState) -> dict:
        if state['iteration'] == 0:
            # First attempt
            prompt = f"Write a Python code for: {state['task']}\nReturn only the code"

        else:
            # Correction attempt - subsequent attempt
            prompt = (
                f"Fix this Python code:\n{state['code']}\n\n"
                f"Errors:\n{state['errors'][-1]}\n\n"
                "Return only the corrected code"
            )


        response = llm.invoke(prompt)

        code = response.content.strip()


        # Clean up markdown code block if present
        if code.startswith("```"):
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[6:]
        
        return {
            "code": code,
            "iteration": state["iteration"] + 1
        }

    
    def validate_code(state: CodeGenState) -> dict:
        code = state['code']

        try:
            # Try to compile the code - not trying LLM to judge their own work
            compile(code, "<string>", "exec") # compile will run code and check the syntax
            return {"success": True}

        except SyntaxError as e:
            return {
                "errors": [f"SyntaxError: {e}"],
                "success": False
            }

        except Exception as e:
            return {
                "errors": [f"Error: {e}"],
                "success": False
            }

    # Route - Core of the loop
    def should_continue(state: CodeGenState) -> Literal["generate", "end"]:
        # if it is compiled
        if state["success"]:
            return "end"
        
        elif state["iteration"] >= state["max_iteration"]: # gurdrails not letting to loop forever
            return "end"

        else:
            "generate"

    
    # clean exit point
    def finalize(state: CodeGenState) -> dict:
        return state

    graph = StateGraph(CodeGenState)

    # Add nodes into graph
    graph.add_node("generate", generate_code)
    graph.add_node("validate", validate_code)
    graph.add_node("finalize", finalize)

    
    # Add edges
    graph.add_edge(START,"generate")

    graph.add_edge("generate","validate")

    graph.add_conditional_edges(
        "validate",
        should_continue,
        {
            "generate":"generate",
            "end":"finalize" # loop back to generate if not successful and under max interation, otherwise it ends everything
        }
    )

    graph.add_edge("finalize", END)


    app = graph.compile()

    print("Self-Correcting Code Generator:\n")

    result = app.invoke({
        "task": "a function that calculates factorial recursively",
        "code": "",
        "errors": [],
        "iteration": 0,
        "max_iteration": 3,
        "success": False
    })


    print(f"Task: {result['task']}")
    print(f"Iterations: {result['iteration']}")
    print(f"Success: {result['success']}")
    print(f"Final code: {result['code']}")





# Iterative Research Agent with Loops and Cycles
# Agent that research a topic, generates a deeper question, then do research again and going more deeper each iteration
# In the end synthesize everything and put together response


# State
class ResearchState(TypedDict):
    topic: str
    findings: Annotated[list[str], operator.add] # concatenate all the findings - not overriding
    questions: list[str]
    iteration: int
    max_depth: int
    summary: str



def iterative_research():
    # Iterative research that goes deeper based on findings
    

    def research(state: ResearchState) -> dict:

        if state["iteration"] == 0:
            query = f"Give me 3 key facts about: {state['topic']}"

        else:
            # go deeper on last generated question
            query = f"Based on these findings:\n{state['findings'][-1]}\n\nGo deeper"
        

        response = llm.invoke(query)
        return {'findings': [response.content]}


    def generate_questions(state: ResearchState) -> dict:
        # look at the latest findings
        response = llm.invoke(
            f"Based on this findings:\n{state['findings'][-1]}\n\n"
            "What's one deeper question to explore? Reply with just the question."
        )

        return {
            "questions": [response.content],
            "iteration": state['iteration'] + 1
        }

    
    def synthesize(state: ResearchState) -> dict:
        all_findings = "\n\n".join(state['findings'])
        response = llm.invoke(
            f"Synthesize these findings into coherent summary:\n\n{all_findings}"
        )

        return {"summary": response.content}

    
    # Core of iteration
    def should_continue(state: ResearchState) -> Literal["research", "synthesize"]:
        if state["iteration"] >= state["max_depth"]:
            return "synthesize"
        
        return "research"


    graph = StateGraph(ResearchState)

    graph.add_node("research", research)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("synthesize", synthesize)


    graph.add_edge(START, "research")
    graph.add_edge("research", "generate_questions")
    graph.add_conditional_edges(
        "generate_questions",
        should_continue,
        {
            "research": "research",
            "synthesize": "synthesize"
        }
    )

    graph.add_edge("synthesize",END)


    app = graph.compile()

    print("Iterative Research")

    # put the initial state for our flow
    result = app.invoke({
        "topic": "quantum computing applications",
        "findings": [],
        "questions": [],
        "iteration": 0,
        "max_depth": 2,
        "summary": ""
    })

    print(f"Topic: {result['topic']}")
    print(f"Iterations: {result['iteration']}")
    print(f"\nFindings collected: {len(result['findings'])}")
    print(f"\nFinal summary:\n{result['summary']}")


if __name__ == "__main__":
    #self_correcting_code()
    iterative_research()