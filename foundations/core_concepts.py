from pydoc import text
from re import M
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate  # construct our prompt
from langchain_core.output_parsers import StrOutputParser # output string
from langchain.chat_models import init_chat_model


load_dotenv()


def demo_basic_chain():
    """Demo a basic chain using LCEL and Runnables"""

    # Component 1: Define the prompt template using LCEL
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer in one sentence: {question}"
    )


    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

    # To parse output
    parser = StrOutputParser()

    # compose with pipe operator using lcel
    chain =  prompt | model | parser

    # Execute the chain with an input - pass question here
    result = chain.invoke({"question": "What is LangChain?"})

    print(f"Response: {result}")

    return chain



def demo_batch_execution():
    """Demo batch execution of a chain."""
    prompt = ChatPromptTemplate.from_template(
        "Transalte to French: {text}"
    )

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    parser = StrOutputParser()

    chain = prompt | model | parser

    # Batch - run with multiple inputs
    # construct inputs as list of objects
    inputs = [{"text":"Hello, how are you?"}, 
              {"text": "What is your name?"},
              {"text": "Where is the nearest restaurant?"}]

    result = chain.batch(inputs) #batch execution for multiple input

    # we can loop through
    for text in zip(inputs, result):
        print(f"Input: {text[0]['text']} => Output: {text[1]}")



def demo_streaming():
    """Demo streaming for real-time output."""
    prompt = ChatPromptTemplate.from_template(
        "Write about a haiku about: {topic}"
    )

    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        streaming=True
    )

    parser = StrOutputParser()


    chain = prompt | model | parser

    # Streaming - run single input
    #input_data = {"topic":" a brave knight"}

    print("Streaming output: ")
    for chunk in chain.stream({"topic":"nature"}):
        print(chunk, end="", flush=True)

    print() # for new line after streaming



def demo_schema_inspection():
    """Demo input/output schema inspection."""
    prompt = ChatPromptTemplate.from_template(
        "Summarize the following text: {text}"
    )


    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    # Inspect input and output schemas
    input_schema = chain.input_schema.model_json_schema()
    output_schema = chain.output_schema.model_json_schema()

    # we see how langchain composes internally 
    print(f"Input Schema: {input_schema}")
    print(f"Output schema: {output_schema}")


def exercise_first_chain():

    prompt = ChatPromptTemplate.from_template(
        "Create a marketing tagline for a product '{product}' targeting '{audience}'"
    )


    model = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7
    )

    parser = StrOutputParser()

    chain = prompt | model | parser

    result = chain.invoke({"product": "AI Course", "audience": "developers"})
    print(f"Marketing Tagline: {result}")



def new_way():
    """The univrsal way to initialize a model is to use init-chat-model"""
    chat_model = init_chat_model("gpt-4o-mini", 
                                temperature = 0.7,
                                max_tokens = 1500)

    

if __name__ == "__main__":
    #demo_basic_chain()
    #demo_batch_execution()
    #demo_streaming()
    #demo_schema_inspection()
    exercise_first_chain()