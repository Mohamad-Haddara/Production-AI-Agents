"""
Full LangGraph Step-by-Step Workflow.

Complete Walkthrough of a graph construction.

Build our first full LangGraph step by step. This allow us to build a conversation graph step by step.
Talk about: State, Nodes, Edges
"""

from typing import Annotated
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from numpy import test
from openai.types.responses.response import Conversation
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage,SystemMessage
import operator
from dotenv import load_dotenv


load_dotenv()



class ConversationState(TypedDict):
    messages: Annotated[list, operator.add]
    sentiment: str
    response_count: int


def create_conversation_graph():

    llm = init_chat_model(model="gpt-4o-mini")


    # 1. Define Node functions
    def analyze_sentiment(state: ConversationState) -> dict:
        """Analyze the sentiment of the last message"""
        last_message = state["messages"][-1]

        response = llm.invoke([
            SystemMessage(content="Classify the sentiment as: positive, negative, or neutral. Reply with just the world"),
            HumanMessage(content=last_message)
        ])

        return {"sentiment": response.content.lower().strip()}

    

    def generate_response(state: ConversationState) -> dict:
        """Generate appropriate response based on sentiment."""

        sentiment = state["sentiment"]
        last_message = state["messages"][-1]


        system_prompts = {
            "positive":"Respond enthusiastically and build on their positive energy",
            "negative": "Respond empathetically and offer support",
            "neutral": "Respond helpfully and informatively"
        }


        prompt = system_prompts.get(sentiment, system_prompts["neutral"]) # get value for key

        reponse = llm.invoke([
            SystemMessage(content=prompt),
            HumanMessage(content=last_message)
        ])

        return {
            "messages": [f"AI:{reponse.content}"],
            "response_count":1
        }


    # Create a graph
    graph = StateGraph(ConversationState)

    # Add node
    graph.add_node("analyze_sentiment", analyze_sentiment)
    graph.add_node("generate_response", generate_response)

    # Add edges
    graph.add_edge(START, "analyze_sentiment")
    graph.add_edge("analyze_sentiment", "generate_response")
    graph.add_edge("generate_response", END)

    app = graph.compile()

    return app


# Instantiate our app
def conversation():
    app = create_conversation_graph()

    # Simulate a conversation
    test_messages = [
        "I just got promoted at work! I'm so excited",
        "My computer crashed and I lost all my work ...",
        "What's weather like today?"
    ]


    print("Conversation Graph Demo:\n")

    for msg in test_messages:
        result = app.invoke({
            "messages": [f"Human: {msg}"],
            "sentiment":"",
            "response_count":0
        })

    
    print(f"Input: {msg}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Response: {result['messages'][-1]}")
    print("-"*40)

if __name__ == "__main__":
    conversation()