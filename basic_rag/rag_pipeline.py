from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import tempfile
from langchain.chat_models import init_chat_model
from typing import List




load_dotenv()

# Initialize embedding model
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")


# Create knowledge base
def create_kb():
    """Create a vector store from knowledge base."""


    # split knowledge base into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )


    doc = Document(page_content = KNOWLEDGEBASE ,
                    metadata = {"source": "langchain_knowledge_base.md"})


    chunks = splitter.split_documents([doc])


    # Create vector store from the chunks
    vector_store = Chroma.from_documents(
        documents= chunks,
        embedding=embeddings_model,
        persist_directory=tempfile.mkdtemp(),
    )

    return vector_store



# Create basic RAG
def demo_basic_rag():
    # create a knowledge base
    vector_store = create_kb()

    retriever = vector_store.as_retriever(search_type = "similarity", search_kwargs = {"k":2})

    llm = init_chat_model(
        model= "gpt-4o-mini",
        temperature = 0.2
    )


    # RAG Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based only on the following context:

        {context}

        Question: {question}

        Answer:

        Make sure to answer ONLY based on context, and if you don't know the answer, just say "I don't know"
        """
    )

    # Format retrieved docs
    def format_docs(docs):
        return "\n\n".join([docs.page_content for doc in docs])

    
    # Rag Chain - LCL pipe syntax
    rag_chain = (
        {"context": retriever | format_docs, "question":RunnablePassthrough()} #question will remain unchanged
        | prompt
        | llm
        | StrOutputParser()
    )


    # Test the RAG chain
    # Test
    questions = [
        "What is LangChain?",
        "Who created LangChain?",
        "What is LangGraph used for?"
    ]


    print("Basic RAG Demo:\n")
    for q in questions:
        answer =rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")


# RAG with Resources
def demo_rag_with_sources():
    vector_store = create_kb()

    retriever = vector_store.as_retriever(
        search_kwargs = {"k":3},
    )

    llm = init_chat_model(model="gpt-4o-mini", temperature=0)

    # Create prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the context below. Include which sources I used.

        Context:
        {context}

        Question: {question}

        Answer (include sources):
        """
    )

    def format_docs_with_sources(docs):
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get('source', 'unknown')
            formatted.append(f"[{i+1}] {source}:\n{doc.page_content}")
        
        return "\n\n".join(formatted)

    rag_chain = (
        {
            "context": retriever | format_docs_with_sources,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG with resources:\n")
    answer = rag_chain.invoke("What are the core components of LangChain?")
    print(f"Q: What are the components?\n")
    print(f"A: {answer}")


# RAG with Fallback - RAG can handle unknown question
def demo_rag_with_fallback():

    vector_store = create_kb()
    llm = init_chat_model(model="gpt-4o-mini", temperature=0)
    retriever = vector_store.as_retriever(search_kwargs = {"k":2})

    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based ONLY on the following context.
        If the answer is not in the context, respond with: "I don't have information about that in my knowledge base."

        Context:
        {context}

        Question: {question}

        Answer:
        """
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
    )


    questions = [
        "What is the pricing for langsmith?", # In knowledge base
        "What is the stock price of OpenAI?", # Not in knowledge base
        "How do I deploy LangChain to AWS", # Not in knowledge base
    ]

    for q in questions:
        answer = rag_chain.invoke(q)
        print(f"Q: {q}")
        print(f"A: {answer}\n")

# RAG with structured outputs - I will use pydantic - we need output to in structured way and we want RAG to be grounded to document
def demo_structured_rag():
    """RAG with structured output."""

    vector_store = create_kb()
    retriever = vector_store.as_retriever(search_kwargs = {"k":3})
    llm = init_chat_model(model="gpt-4o-mini", temperature = 0)

    
    class RAGResponse(BaseModel):
        """Structured RAG response."""
        asnwer: str = Field(description="The answer to the question")
        confidence: str = Field(description="high, medium, low")
        sources_used: List[str] = Field(description="List of sources referenced")
        follow_up: str = Field(description="Suggested follow-up question")

    structured_llm = llm.with_structured_output(RAGResponse)

    prompt = ChatPromptTemplate.from_template(
        """
        Based on the context below, asnwer the question.

        Context:
        {context}

        Question: {question}

        Provide a structured response.
        """
    )


    def format_docs(docs):

        return "\n\n".join(f"[{doc.metadata.get('source', 'unknown')}]: {doc.page_content}" 
                        for doc in docs)


    rag_chain = (
        {"context": retriever|format_docs, "question": RunnablePassthrough()}
        | prompt
        | structured_llm
    )
    
    print("Structured RAG Demo:\n")
    # it is structured output, it is dictionary
    result = rag_chain.invoke("What is LangGraph?")

    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
    print(f"Sources: {result.sources_used}")



# Hands-on RAG pipeline exercise

if __name__ == "__main__":
    #demo_basic_rag()
    #demo_rag_with_fallback() --> check if the response for context of the question not in knowledge base, does it return we dont have it in knowledge base