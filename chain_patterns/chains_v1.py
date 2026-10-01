"""
Understanding Chains in LangChain
LCEL patterns, composition, and debugging
"""



from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model # recommended to initialize model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough, 
    RunnableLambda,
    RunnableBranch
    )



load_dotenv()


model = init_chat_model(model="gpt-4o-mini", temperature = 0)



# Sequential Chain
def demo_basic_chain():
    prompt = ChatPromptTemplate.from_template("Summarize the following text in one sentence: {text}")

    parser = StrOutputParser()

    # Runnable
    chain = prompt | model | parser

    result = chain.invoke({"text": "LangChain is a framework for developing applications powered by language models."})


    print(f"Summary: {result}")



# Parallel Chain
def demo_parallel_chain():
    """Run multiple chains in parallel."""

    # Define individual chains
    summarize_prompt = ChatPromptTemplate.from_template("Summarize in two sentences: {text}")

    keyword_prompt = ChatPromptTemplate.from_template("Extract 5 keywords in the following text: {text}\n Return as comma-separated")


    sentence_prompt = ChatPromptTemplate.from_template("What is the sentiment of the following text? {text}")

    parser = StrOutputParser()

    # Parallel Execution - Come in as dictionary
    analysis_chain = RunnableParallel(
        summary = summarize_prompt | model | parser,
        keywords = keyword_prompt | model | parser,
        sentiment = sentence_prompt | model |parser
    )

    # Text will be used as input
    text = """
    The new AI features are absolutely incredible! Users are loving the faster response times and improved accuracy.
    However, some have noted that the pricing could be more competitive. Overall, the product lunch has been a massive 
    sucess with record-banking adoption rates.

    """

    results = analysis_chain.invoke({"text": text})
    
    print("Analysis Results:")
    print("Parallel Analysis Results:")
    print(f"    Summary: {results['summary']}")
    print(f"    Keywords: {results['keywords']}")
    print(f"    Sentiment: {results['sentiment']}")


# PassThrough Runnable - used for RAG system
def demo_passthrough_chain():
    """PassThrough Chain pattern"""

    prompt = ChatPromptTemplate.from_template(
        "Original question: {question}\n"
        "Context: {context}\n"
        "Answer the question based on the context."
    )


    def fake_retriever(input_dict):
        return "Langchain was created by Moe in 2022."

    # Chain will retrieve context and question in parallel, then transform structure
    chain = (
        RunnableParallel(
        context = RunnableLambda(fake_retriever), question = RunnablePassthrough(),
    ) 
    | RunnableLambda(
        lambda x: {"context": x["context"], 
                    "question": x['question']['question']}
    )
    | prompt 
    | model
    | StrOutputParser()
    
    )

    result = chain.invoke({"question": "Who created langchain?"})
    print(f"Answer: {result}")




# Uses 2 LLM per request
# 1) classifier ~ determine route
# 2) select chain to run
def demo_chain_branching():
    """chain brunching (conditional)"""
    
    code_prompt = ChatPromptTemplate.from_template(
        "You are a coding expert. Help with: {input}"
    )

    general_prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant. Answer: {input}"
    )

    # Classifier
    classifier_prompt = ChatPromptTemplate.from_template(
        f"Classify this as 'code' or 'general: {input}\nReturn onlt the classification"
    )


    classifier = classifier_prompt | model | StrOutputParser()


    # Branching chain based on classification
    def is_code_question(input_dict):
        classification = classifier.invoke(input_dict)
        return "code" in classification.lower()

    # Practice RunnableBranch class
    branch = RunnableBranch(
        (is_code_question, code_prompt | model | StrOutputParser()),
        general_prompt | model | StrOutputParser(), # default branch
    )

    # Test
    questions = [
        "How do I write a for loop in Python?", 
        "What's the weather like today?"
    ]

    for q in questions:
        result = branch.invoke({"input": q})
        print(f"Q: {q}")
        print(f"A {result[:100]}...\n")


def demo_debugging():
    prompt = ChatPromptTemplate.from_template("Say hello to {name}")

    chain = prompt | model | StrOutputParser()


    # Method 1 - Get configuration
    print(f"Chain input schema: {chain.input_schema.model_json_schema()}")
    print(f"Chain output schema: {chain.output_schema.model_json_schema()}")


    # Method 2 - Use with_config for tracing
    result = chain.with_config(
        run_name = "greeting_chain"
    ).invoke({"name": "Alice"})


    print(f"Greeting: {result}")

    # Method 3: Inspect intermediate steps
    # Using RunnableLambda for logging

    def log_step(x, step_name=""):
        print(f"[{step_name}] {type(x).__name__}: {str(x)[:100]}")
        return x


    debug_chain = (
        prompt
        | RunnableLambda(lambda x: log_step(x, "after_prompt"))
        | model
        | RunnableLambda(lambda x: log_step(x, "after_model"))
        | StrOutputParser()
    )


    print("\nDebug chain execution:")
    result = debug_chain.invoke({"name":"Debug"})
    print(f"Greeting: {result}")



if __name__ == "__main__":

    #demo_basic_chain()
    #demo_parallel_chain()
    #demo_passthrough_chain()
    #demo_chain_branching()
    demo_debugging()