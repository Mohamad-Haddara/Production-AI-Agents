from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Literal, Type
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage, SystemMessage
import operator
from dotenv import load_dotenv



load_dotenv()


llm = init_chat_model("gpt-4o-mini", temperature = 0.0)


# Create Router State class
class RouterState(TypedDict):
    query: str
    query_type: str
    response: str


def basic_routing():

    def classify_query(state:RouterState) -> dict:

        response = llm.invoke([
            SystemMessage("Classify this query as 'question', 'command', or 'statement'. Reply with just the word."),
            HumanMessage(state['query']),
        ])

        return {"query_type": response.content.lower().strip()}

    def handle_question(state: RouterState) -> dict:
        response = llm.invoke(f"Answer this question: {state['query']}")
        return {"response": f"[Answer] {response.content}"}

    def handle_command(state: RouterState) -> dict:
        return {"response": f"[Executing] I'll help you with: {state['query']}"}

    def handle_statement(state: RouterState) -> dict:
        return {"response": f"[Acknowledged] Thanks for sharing: {state['query']}"}


    # Actual Route by Type
    def route_by_type(state: RouterState) -> Literal["question", "command", "statement"]:
        qt = state["query_type"]

        if "question" in qt:
            return "question"
        
        elif "command" in qt:
            return "command"

        else:
            return "statement"
    

    # Create a graph
    graph = StateGraph(RouterState)


    # Add node to the graph
    graph.add_node("classify", classify_query)
    graph.add_node("handle_question", handle_question)
    graph.add_node("handle_command", handle_command)
    graph.add_node("handle_statement", handle_statement)
    


    # Add edges
    graph.add_edge(START, "classify")
    # Add conditional edges
    graph.add_conditional_edges(
        "classify", # source
        route_by_type, # routing function to determine which edges to take based on the source
        # Map: route_key:route_name~ actual node in graph
        {
            "question": "handle_question",
            "command": "handle_command",
            "statement": "handle_statement"
        }
    )


    graph.add_edge("handle_question", END)
    graph.add_edge("handle_command", END)
    graph.add_edge("handle_statement", END)


    app = graph.compile()

    # example usage
    queries = [
        "What is the capital of France",
        "Send an email to John",
        "I love programming"
    ]

    
    for query in queries:
        result = app.invoke({"query": query})
        print(f"Query: {query}")
        print(f"Type: {result['query_type']}")
        print(f"Response: {result['response']}") 
        print("-"*40)



# Conditional Looping
# Edge can loop back


class QualityState(TypedDict):
    content: str
    quality_score: int
    feedback: str
    final_content: str
    iteration: int


def conditional_loop():

    def evaluate_quality(state: QualityState) -> dict:

        response = llm.invoke([
            SystemMessage("Rate this content quality from 1-10. Reply with just the number."),
            HumanMessage(state['content'])

        ])

        try:
            score = int(response.content.strip())
        
        except:
            score = 0

        return {"quality_score": score}

    
    def improve_content(state:QualityState) -> dict:

        response = llm.invoke(f"Improve this content to be more engaging and clear:\n\n{state['content']}")

        return {
            "content": response.content,
            "iteration": state["iteration"] + 1
        }



    def finalize_content(state: QualityState) -> dict:
        return {
            "final_content": state['content'],
            "feedback": f"Approved after {state['iteration']} iteration with score {state['quality_score']}"
        }

    
    # define the actual loop ~ Literal Routing
    def should_continue(state: QualityState) -> Literal["improve", "finalize"]:
        if state["quality_score"] >= 7:
            return "finalize"
        
        elif state["iteration"] >= 3:
            return "finalize" # Maz iteration
        
        else:
            return "improve"


    graph = StateGraph(QualityState)

    
    graph.add_node("evaluate", evaluate_quality)
    graph.add_node("improve", improve_content)
    graph.add_node("finalize", should_continue)


    graph.add_edge(START, "evaluate")


    graph.add_conditional_edges(
        "evaluate",
        should_continue,
        {"improve": "improve", "finalize": "finalize"}
    )

    
    graph.add_edge("improve","evaluate") # Loop back, by adding edge
    graph.add_edge("finalize", END)



    app = graph.compile()

    # Example usage
    print(f"Conditional Loop:")

    result = app.invoke(
        {
            "content": "AI is cool",
            "quality_score": 0,
            "feedback":"",
            "final_content": "",
            "iteration":0
        }
    )
    

    print(f"Original: AI is cool")
    print(f"Final: {result['final_content'][:200]}...")
    print(f"Feedback: {result['feedback']}")


# Multipath Routing
def multi_path_routing():
    class TaskState(TypedDict):
        task: str
        urgency: str
        complexity: str
        handler: str
        result: str

    
    
    # Analyzer
    def analyze_task(state: TaskState):
        # Analyze urgency
        urgency_response = llm.invoke(
            f"Is this task urgent? Reply 'urgent' or 'normal'.\nTask: {state['task']}"
        )

        # Analyze complexity
        complexity_response = llm.invoke(
            f"Is this task complex? Reply 'complex' or 'simple'.\nTask: {state['task']}"
        )



        return {
            "urgency": urgency_response.content.lower().strip(),
            "complexity": complexity_response.content.lower().strip()
        }



    # Handler
    def urgent_complex_handler(state: TaskState) -> dict:
        return {
            "handler": "Senior Team",
            "result": "Escalated to senior team for immediate action",
        }

    
    def urgent_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Quick Response",
            "result": "Handled immediately by available agent"
        }


    def normal_complex_handler(state: TaskState) -> dict:
        return {
            "handler":"Specialist",
            "result": "Assigned to specialist for"
        }


    def normal_simple_handler(state: TaskState) -> dict:
        return {
            "handler": "Standard",
            "result": "Added to standard queue"
            }

    

    # Routing function
    def route_task(state: TaskState) -> str:
        is_urgent = "urgent" in state["urgency"]
        is_complex = "complex" in state["complexity"]

        if is_urgent and is_complex:
            return "urgent_complex"

        elif is_urgent:
            return "urgent_simple"

        elif is_complex:
            return "normal_complex"
        
        else:
            return "normal_simple"

    

    graph = StateGraph(TaskState)


    graph.add_node("analyze", analyze_task)
    graph.add_node("urgent_complex", urgent_complex_handler)
    graph.add_node("urgent_simple", urgent_simple_handler)
    graph.add_node("normal_complex", normal_complex_handler)
    graph.add_node("normal_simple", normal_simple_handler)


    graph.add_edge(START, "analyze")

    graph.add_conditional_edges(
        "analyze", # Source
        route_task, # Function
        {   # Mapping
            "urgent_complex": "urgent_complex",
            "urgent_simple": "urgent_simple",
            "normal_complex": "normal_complex",
            "normal_simple": "normal_simple", 
        },
    )

    for node in ["urgent_complex", "urgent_simple", "normal_complex", "normal_simple"]:
        graph.add_edge(node, END)




    app = graph.compile()

    print("\nMulti-Path Routing:\n")

    tasks = [
        "Server is down! Need immediate fix!",
        "Update the documentation for the API",
        "Redesign the entire database schema",
        "Fix the typo on the homepage"
    ]

    for task in tasks:
        result = app.invoke({"task": task})
        print(f"Task: {task}")
        print(f"Urgency: {result['urgency']} | Complexity: {result['complexity']}")
        print(f"Handler: {result['handler']}")
        print(f"Result: {result['result']}")
        print("-"*40)









if __name__ == "__main__":
    #basic_routing()
    #conditional_loop()
    multi_path_routing()