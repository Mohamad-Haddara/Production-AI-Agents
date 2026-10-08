"""
LangGraph Core Concepts
StateGraph, nodes, edges, and basic patterns
"""

from ast import List, Mult
from email import message
from langchain.chat_models import init_chat_model
from langchain_community.llms.sagemaker_endpoint import ContentHandlerBase
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import operator
from dotenv import load_dotenv


load_dotenv()



# Graph has nodes that do work and it has edges that connect these nodes

# Basic state - which we can use in our nodes
# define class and inheritance
class SimpleState(TypedDict):
    user_input: str
    output: str
    step: int # step are tracked as integers


def demo_simple_graph():
    # First we need to define node functions - remember 3 main pillars: states - nodes - edges (connectors)
    # Node ~ is just a function - it should has a state

    def process(state: SimpleState) -> dict:
        # simple processing logic, for demo purposes
        return {"output": state["user_input"].upper(), "step":state["step"] + 1}

    
    # Second step create a graph - Then pass the context ~ which is the state
    graph = StateGraph(SimpleState)


    # Third step is to Add nodes (node name, actual function)
    graph.add_node("process", process)

    # Fourth step is to Add edges (connector) ~ START --> process node 
    graph.add_edge(START, "process")    # A is connected to B
    graph.add_edge("process", END)


    # Execute graph / compile
    app = graph.compile()


    # run app--> inside invoke is the state
    result = app.invoke({"user_input": "hello", "output": "", "step":0})

    print(f"Simple graph result: {result}") # fetch each piece that we have inside the output 
    print(f" Input: {result['user_input']},  Output: {result['output']}, Step: {result['step']}")



    # LangGraph allow us to create visual
    # Save as PNG
    # png_bytes = app.get_graph().draw_mermaid_png()
    # with open("graph.png", "wb") as f:
    #     f.write(png_bytes)

    # print("\nGraph saved to graph.png")



    """
    3 Steps to memorize:
        1) Define function node
        2) Create StateGraph and we pass object inside it to track everything
        3) Add nodes (Goes From START --> Process --> END) and edges
        4) Execute and compile
    """


"""
Lesson 2: Understanding 
State accumulates values instead of overwriting or deleting things.
This is important because of context, the state has to be source of truth to our workflow. It has to be traceable.

"""

# === State with Reducers ===
class AccumulatingState(TypedDict):
    messages: Annotated[list[str], operator.add] # concatenate list of items
    count: Annotated[int, operator.add] # counts sum when merged - V.I bcz it tell langgraph don't override these fields, but run function and combine the old with new



def demo_accumulating_state():

    # Define 2 node
    def step_one(state: AccumulatingState) -> dict:
        return {
            "messages": ["Step 1 executed"],
            "count": 1
        }

    def step_two(state: AccumulatingState) -> dict:
        return {
            "messages": ["Step 2 executed"],
            "count": 1
        }


    # Create StateGraph object
    graph = StateGraph(AccumulatingState)




    # Add nodes into the graph
    graph.add_node("step_one", step_one)
    graph.add_node("step_two", step_two)

    graph.add_edge(START, "step_one")
    graph.add_edge("step_one", "step_two") # connection between step 1 and step 2

    graph.add_edge("step_two", END)

    # compile
    app = graph.compile()
    result = app.invoke({
                            "messages": ["Initial message"], 
                            "count": 0}
                            )

    print("\nAccumulating State Result:")
    print(f"    Messages: {result['messages']}")
    print(f"    Count: {result["count"]}")

       # Visualize the graph
    print("\n--- Mermaid Graph ---")
    


    # save as PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph_2.png", "wb") as f:
        f.write(png_bytes)

    """
    First we passed the Intial message but the count is 0
    Then execute step 1 with count is 1
    Then step 2 is executed and count 1 ~ count is incremented

    V.I topic in langgraph -  without reducer ~ each node will override the fields, which means as we go down (flow down the graph),
    the last step is not going to understand what happened in the first one because it is overriding

    With reducer ~ each node is preserved.
    """




# === Message State (Common Pattern) ===
"""
Message State - The Chat Pattern

Most of the apps that are going to be building with LangGraph ~ chatapps

Message State ~ I will see it a lot in chatbots and agents, they need messages


"""



from langgraph.graph import add_messages

class MessageState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages] # messages concatenate (replace operator.add)



def demo_message_state():
    llm = init_chat_model("gpt-4o-mini", temperature = 0)

    # define the node - state will always has something ~ preserve all the state and messages
    def chat_node(state: MessageState) -> dict:
        response = llm.invoke(state["messages"])
        return {'messages': [response]}


    def chat_node_2(state: MessageState) -> dict:
        response = llm.invoke(state["messages"])
        return {'messages': [response]}

    # Create graph
    graph = StateGraph(MessageState)

    # Add node to the graph
    graph.add_node("chat_node", chat_node)

    # Add edge
    graph.add_edge(START, "chat_node")
    graph.add_edge("chat_node", END)

    # compile
    app = graph.compile()

    result = app.invoke({"messages": [HumanMessage(content="Say Hello in French")]})

    print("\nMessage State Result:")
    for msg in result["messages"]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"    {role}: {msg.content}")

    """
    We went from Human --> AI
    Very simple and very important

    Key takeaway:
        We are using message state + LLM node (which are the building block of all chat agents) 
        state is always connected

    """


# === Multi-node Graph ===
"""
Multi-Node Pipelines - Chaining LLM Calls
(more complex as real-world ~ bunch of nodes working together to create a graph)

Put a graph with multiple processing nodes.
"""

# Define state class
class MultiStepState(TypedDict):
    user_input: str
    analyzed: str
    enhanced: str
    final: str



def demo_multi_node_graph():

    llm = init_chat_model("gpt-4o-mini", temperature = 0)
    
    # Define the node function ~ more than one node
    def analyze_node(state: MultiStepState) -> dict:
        response = llm.invoke([HumanMessage(content=f"Analyze the following input and summarize it in one sentence: {state['user_input']}")])
        return {"analyzed": response.content}

    def enhance(state: MultiStepState) -> dict:
        response = llm.invoke([HumanMessage(content=f"Take the following analysis and enhance it with more details: {state['analyzed']}")])
        return {"enhanced": response.content}  

    def finalize(state: MultiStepState) -> dict:
        response = llm.invoke([HumanMessage(content=f"Take the following enhanced and finalize it into concise summary: {state['enhanced']}")])
        return {"final": response.content}


    # lets put all together ~ graph need to have state
    graph = StateGraph(MultiStepState)

    # Add nodes into graph
    graph.add_node("analyze_node", analyze_node)
    graph.add_node("enhance_node", enhance)
    graph.add_node("final_node", finalize)

    # Add edges
    graph.add_edge(START, "analyze_node")
    graph.add_edge("analyze_node", "enhance_node")
    graph.add_edge("enhance_node","final_node")
    graph.add_edge("final_node", END)


    app = graph.compile()

    result = app.invoke({"user_input": "Artificial Intelligence"})

    # state have all these feilds
    print("\nMulti-Node Graph Result:")
    print(f"    Input: {result["user_input"]}")
    print(f"    Analyzed: {result['analyzed'][:100]}...")
    print(f"    Enhanced: {result['enhanced'][:100]}...")
    print(f"    Final: {result['final']}")


    # visualize the graph
    print("\n --- Mermaid Graph ---")

    # Save as PNG
    png_bytes = app.get_graph().draw_mermaid_png()
    with open("graph_3.png", "wb") as f:
        f.write(png_bytes)

    """
    Key takeway:
        We an chain nodes for multi-step reasoning 
        Most important thing that each node reads previous output and writes its own

        Each node is independent but it knows what is happening previously
    """



# Exercise

class MultiStepState(TypedDict):
    topic: str
    questions: str
    answer: str
 

def exercise_first_langgraph():
    """
    EXERCISE: Create a LangGraph that:
    1. Takes a topic as input
    2. Node 1: Generate 3 questions about the topic
    3. Node 2: Answers one of the questions
    4. Returns both questions and answer
    """
    # Define llm 
    llm = init_chat_model(model="gpt-4o-mini", temperature = 0)

    # Define node function
    def generate_question(state: MultiStepState) -> dict:
        response = llm.invoke([HumanMessage(content=f"Generate 3 questions about the topic: {state["topic"]}. Format: numbered list")])
        return {"questions": response.content}

    def answer_one_question(state: MultiStepState) -> dict:
        response = llm.invoke([HumanMessage(content=f"Answer the first question from this list:\n: {state['questions']}")])
        return {
            "answer": response.content,
        }

    # Create a state
    graph = StateGraph(MultiStepState)

    # Add Node into the graph
    graph.add_node("generate_questions", generate_question)
    graph.add_node("answer_question", answer_one_question)

    graph.add_edge(START, "generate_questions")
    graph.add_edge("generate_questions", "answer_question")
    graph.add_edge("answer_question", END)
    
    app = graph.compile()

    result = app.invoke({"topic": "The future of renewable energy"})
    
    # state have all these feilds
    print("\nExercise Multi-Node Graph Result:")
    print(f"    Input: {result["topic"]}")
    print(f"    questions: {result['questions']}")
    print(f"    Answer: {result['answer']}")
    
"""
These steps I will always use when using LangGraph.
We have pillars --> State, Nodes, Edges

"""



    




if __name__ == "__main__":
    #demo_simple_graph()
    #demo_accumulating_state()
    #demo_message_state()
    #demo_multi_node_graph()
    exercise_first_langgraph()