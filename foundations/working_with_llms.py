"""
working with LLMs in LangChain

Multiple providers, configuration, streaming, and cost optimization
"""

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from openai import max_retries
from openai.types import chat
from typing import Any

load_dotenv()


def demo_init_chat_model():
    chat_model = init_chat_model(
        model="gpt-4o-mini",
        model_provider = "openai",
        # Parameters to configure our model
        temperature = 0.7,
        streaming = True,
        max_retries = 3
    )


    response = chat_model.invoke("What is the capital France?")

    print(f"Response is: {response.content}")
    return chat_model



def demo_model_comparison():

    prompt = "Explain recursion in one sentence"

    models = {
        "gpt-4o-mini": init_chat_model(
            model="gpt-4o-mini",
            temperature = 0.7,
            streaming = False
        ),

        "gpt-4o": init_chat_model(
            model="gpt-4o",
            temperature = 0.7,
            streaming = False
        ),
    }


    print(f"Prompt: {prompt}")

    for model_name, model in models.items():
        response = model.invoke(prompt)
        print(f"Response from {model_name}: {response.content}\n")


def demo_message():
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # Using message objects (more control)
    messages = [
        # Wraper -It is impotant to have these class because allow us to add additional information other than content
        SystemMessage(content="You are helpful assistant."),
        HumanMessage(content="What is the capital of France? Answer in one word")
    ]

    # print("Using message objects:")
    # print(f"Message: {messages[0]} | {messages[1]}")

    # I can pass many messages
    response = model.invoke(messages) # return AIMessage
    print(f"Response : {response.content}") 


    # I can call Multi-turn conversation using message objects - I can use messages list and append the response
    messages.append(response) #add model response to the conversation, and added as AIMessage context
    print(messages)

    # Continuation of conversation
    messages.append(HumanMessage(content="What about tomorrow?"))

    print("\nMulti-turn conversation:")
    response = model.invoke(messages)
    print(f"\nFollow-up response: {response.content}")


# Multi model setup
def exercise_multi_model():
    """
    1. Takes a question and a list of model names
    2. Gets response from all models
    3. Returns a dict of {model_name: response}
    """



    def get_responses(question: str, model_names: list[str]) -> dict[str, Any]:
        responses = {}

        for model_name in model_names:
            model = init_chat_model(
                model = model_name,
                temperature = 0.7,
                streaming = False
            )

            response = model.invoke(question)
            responses[model_name] = response.content

        return responses # dict

    
    result = get_responses("What is AI", ["gpt-4o", "gpt-4o-mini"])
    for model, response in result.items():
        print(f"Response from: {model}: {response}\n")

if __name__ == "__main__":
    #demo_init_chat_model()
    #demo_model_comparison()
    #demo_message()
    exercise_multi_model()