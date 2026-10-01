"""
Complete RAG system with conversation memory
"""

from importlib import metadata
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import (
    InMemoryChatMessageHistory,
    BaseChatMessageHistory
)

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor

from pydantic import BaseModel, Field
from typing import List, Dict, Optional

from datetime import datetime
from dotenv import load_dotenv



load_dotenv()


# 1. First define data model

# ==================================================
# Data Models
# ==================================================

# Research response ~ Inherit base model
class ResearchResponse(BaseModel):
    """Structured response from the research assistant."""

    answer: str = Field(description="The answer to the question")
    confidence: str = Field(description="High, Medium, or Low based on source quality")
    sources: List[str] = Field(description="List of source documents used")
    key_quotes: List[str] = Field(
        description="Relevant quotes from sources", default=[]
        )
    follow_up_questions: List[str] = Field(description="Suggested follow up question", default=[])


#  Every RAG system needs: Embedding Model, Text Splitter, Vector Store (Encapsulate)

# ==================================================
# Research Assistant Class (core of RAG system)
# ==================================================

class AIResearchAssistant:
    """
    AI Research Assistant with document ingestion and retrieval.
    """

    # Constructor - Instantiate embedding model, recursive splitter and vector database
    def __init__(
        self,
        persistant_directory : str = "./research_db",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):


        self.presistant_directory = persistant_directory

        # 1. Embeddings - turns text into vectors
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        # 2. Splitter - break big docs into chunks
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            separators=["\n\n", "\n", ".", " ", ""]
        )


        # 3. Vector store - stores and searches embeddings
        self.vectorstore = Chroma(
            persist_directory=persistant_directory,
            embedding_function= self.embeddings, 
            collection_name="research_docs",
        )

        # 4. Initialize llm
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        # 5. Initialize session store - to store history conversation - allow each user session to get its own history
        self.session_store : Dict[str, InMemoryChatMessageHistory] = {}
  
    # Add document method - allow us to ingest documents
    def add_documents(
        self,
        documents: List[Document],
        source_name: Optional[str] = None
    ) -> int:
        """
        Add documents to the research database.
        """


        # Tag with soruce name
        if source_name:
            for doc in documents:
                doc.metadata["source"] = source_name

        
        # Split into chunks
        chunks = self.splitter.split_documents(documents)


        # Timestamp each chunk
        for chunk in chunks:
            chunk.metadata['indexed_at'] = datetime.now().isoformat()


        # store in vector DB
        self.vectorstore.add_documents(chunks)


        return len(chunks)

    # Ingest raw text
    def add_text(
        self,
        text: str,
        source: str,
        metadata: dict = None
    ) -> int:
        """Add single text string as a document."""
        doc = Document(
            page_content=text,
            metadata = {'source': source, **(metadata or {})}
        )

        return self.add_documents([doc])


    def add_texts(
        self,
        texts: List[str],
        source: str,
    ) -> int:
        """
        Add multiple text strings from the same source.
        """

        docs = [Document(page_content=t, metadata= {"source":source}) for t in texts]

        return self.add_documents(docs)

    # Inspection methods
    def get_document_count(self) -> int:
        """Get total number of indexed chunks."""
        return self.vectorstore._collection.count()


    def list_sources(self) -> List[str]:
        """List all unique sources in the database."""
        results = self.vectorstore._collection.get()
        sources = set()

        for metadata in results.get("metadatas", []):
            sources.add(metadata["source"])

        return sorted(list(sources))

    # retrieval method -> basic similarity search
    def _build_retriever(self, use_advanced: bool = False):
        """Build a basic similarity retriever."""
        
        # Base: Simple similarity search
        base_retriever =  self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs = {"k":4} # retrieve 4 relevant documents (chunks)
        )

        if not use_advanced:
            return base_retriever

        # Multi-query: LLM generates multiple search queries --> use_advanced = True
        multi_retriever = MultiQueryRetriever.from_llm(
            retriever = base_retriever,
            llm=self.llm
        )

        return multi_retriever



    def _format_docs_for_context(self, docs) -> str:
        """
        Format retrieved documents into a string for the prompt
        
        Goal to turn document object into strings that LLM can actually read
        So our documents will be well formatted
        """
        if not docs:
            return "No relevant documents founds"
        
        formatted = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown")
            formatted.append(f"[Source: {i+1}: {source}]\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted)

    def ask_structured(
        self,
        question: str,
        session_id: str = "default",
        use_advanced: bool = True
    ) -> ResearchResponse:
        """Ask a question and get a structured response."""

        # LLM that returns a Pydantic object instead of a string
        structured_llm = self.llm.with_structured_output(ResearchResponse) # llm will parse through ResearchResponse class

        # Get memory
        history = self._get_session_history(session_id)

        # Retrieve
        retriever = self._build_retriever(use_advanced)
        docs = retriever.invoke(question)
        context = self._format_docs_for_context(docs)
        sources = [ set(d.metadata.get("source", "Unknown") for d in docs)]


        # Step 3: Build the prompt + Add memory + available sources
        prompt = ChatPromptTemplate.from_messages(
            [
            ("system", """You are an AI Research Assistant. 
            Answer questions based ONLY on the provided context documents.
            
            Rules:
                1. Only use information from the context below
                2. If the context doesn't have the answer, say so
                3. Cite which sources you used (e.g According to Source 1...)
                4. Rate our confidence: high, medium, or low 
            """),

            MessagesPlaceholder(variable_name="history"), # do all the heacy lifting -> it expands the actual messages at runtime - first call it will be empty, 3rd call it will have 4 messages in there

            (
                "human",
                """
                Context documents:

                {context}
                
                Available sources: {sources}

                Question: {question}

               Provide a clear answer with source citations. 
                
                """
            )
        
            ]
        )

        chain = prompt | structured_llm


        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "sources": ", ".join(sources),
                "history": (
                    history.messages[-10:]
                    if hasattr(history, "messages")
                    else history[-10:]
                ),
            }
        )

        # Save to memory (store just the answer text)
        history.add_message(HumanMessage(content=question))
        history.add_message(AIMessage(content=response))


        return response



    # toggle ~ flag (so if we want to use advanced retriever or not)
    def ask(self, question: str, session_id: str = "default", use_advanced: bool = True) -> str:  # -> str is fine but not ideal, bcz it is difficult to work with it to extract --> Use ResearhResponse - good for strucutre response
        """Ask a question against the research documents."""

        # Before we retrieve -> get session history
        history = self._get_session_history(session_id)

        # Use basic or advanced retriever
        # Step 1: Retrieve relevant chunks
        retriever = self._build_retriever(use_advanced)
        docs = retriever.invoke(question)

        # Step 2: Format into context string
        context = self._format_docs_for_context(docs)
        
        # Step 3: Build the prompt + Add memory
        prompt = ChatPromptTemplate.from_messages(
            [
            ("system", """You are an AI Research Assistant. 
            Answer questions based ONLY on the provided context documents.
            
            Rules:
                1. Only use information from the context below
                2. If the context doesn't have the answer, say so
                3. Cite which sources you used (e.g According to Source 1...)
                4. Rate our confidence: high, medium, or low 
            """),

            MessagesPlaceholder(variable_name="history"), # do all the heacy lifting -> it expands the actual messages at runtime - first call it will be empty, 3rd call it will have 4 messages in there

            (
                "human",
                """
                Context documents:

                {context}
                
                Question: {question}

               Provide a clear answer with source citations. 
                
                """
            )
        
            ]
        )

        # Step 4: Build and run the chain
        chain = prompt | self.llm | StrOutputParser()
        

        response = chain.invoke(
            {
                "context": context,
                "question": question,
                "history": history.messages[-10:] # last 10 messages for context
            }
        )

        # Save this exchange to memory (save this Q&A to history)
        history.add_message(HumanMessage(content=question)) # add human message
        history.add_message(AIMessage(content=response)) # Add AI message



        return response

    def clear_session(self, session_id: str):
        """Clear conversation history for a session."""
        if session_id in self.session_store:
            self.session_store[session_id].clear()
            print(f"Cleared session: {session_id}")


    def get_session_history_display(self, session_id:str) -> list:
        """Get conversation history as readable dict."""
        if session_id not in self.session_store:
            return []

        return [
            {
                "role": "human" if isinstance(m, HumanMessage) else "assistant",
                "content": m.content
            }
            for m in self.session_store[session_id].messages
        ]

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """
        Get or create session history. 
        
        All memory is a list of messages and getting it into the prompt.

        Add it into prompt
        """
        if session_id not in self.session_store:
            self.session_store[session_id] = InMemoryChatMessageHistory()
        
        return self.session_store[session_id]





def print_search_response(question:str, response: ResearchResponse):
    """Pretty print a structured research response."""

    print(f"\nQ:{question}")
    print(f"\n  Answer: {response.answer}")
    print(f"\n  Confidence: {response.confidence}")
    print(f"\n  Sources: {response.sources}")


    if response.key_quotes:
        print(f"\n  Key Quotes:")
        for q in response.key_quotes:
            


if __name__ == "__main__":
    import shutil
    # Clean start
    shutil.rmtree("./research_db", ignore_errors=True)

    assistant = AIResearchAssistant()

    

    # Sample text to test
    assistant.add_text(
        """
    Retrieval-Augmented Generation, commonly called RAG, is an architecture that combines a large language model with an external knowledge source so that answers are grounded in retrieved documents rather than only in the model's parametric memory. A base LLM has a fixed knowledge cutoff, cannot see private enterprise data, and may hallucinate when asked about facts it never saw during training. RAG addresses these problems by retrieving relevant passages at query time and injecting them into the prompt as context, allowing the model to generate a response that cites and depends on real source material.
 
A RAG pipeline has two main phases: ingestion and query-time retrieval. During ingestion, raw documents such as PDFs, Word files, HTML pages, and spreadsheets are parsed into clean text. Document parsing is often underestimated; tables, multi-column layouts, headers, and scanned pages can easily be corrupted by naive extraction. Tools such as Docling or Unstructured convert complex layouts into structured representations like Markdown, preserving headings and table boundaries. The parsed text is then split into chunks. Chunking strategy strongly affects retrieval quality. Fixed-size chunking with token overlap is simple, but structure-aware chunking that respects section headings, paragraphs, and table rows usually produces more coherent retrieval units. Typical chunk sizes range from 256 to 1024 tokens, with an overlap of 10 to 20 percent to avoid cutting important context at boundaries.
 
Each chunk is converted into a dense vector using an embedding model. The embedding maps semantically similar text to nearby points in a high-dimensional space, so a question and a passage that answers it end up close together even if they share few exact words. Multilingual models such as BGE-M3 are especially useful in bilingual deployments, for example Arabic and English, because they place both languages into a shared vector space. The vectors are stored in a vector database such as Qdrant, Weaviate, Milvus, or pgvector, along with metadata payloads like document ID, section title, page number, language, access permissions, and tenant ID. Metadata is critical because it enables filtering at query time, for instance restricting retrieval to documents the current user is authorized to see.
 
At query time, the user's question is embedded with the same model and a nearest-neighbor search is performed, typically using an approximate index such as HNSW for low latency at scale. Pure dense retrieval, however, can miss exact matches on rare terms, product codes, policy numbers, or acronyms. For this reason, production systems usually implement hybrid search, combining dense vectors with sparse lexical retrieval such as BM25 or learned sparse vectors. The two result lists are merged using Reciprocal Rank Fusion, which scores each document based on its rank in each list rather than on raw scores that live on different scales.
 
The fused candidates are then passed to a reranker, usually a cross-encoder model that reads the query and each candidate passage together and outputs a relevance score. Cross-encoders are slower than bi-encoder embeddings, so they are applied only to the top 20 to 50 candidates, but they significantly improve precision in the final top-k. The highest-ranked chunks are assembled into a context window with source identifiers, and the LLM is prompted to answer strictly from that context and to cite the sources it used. A well-designed prompt also instructs the model to say when the context does not contain the answer, which is one of the most effective ways to reduce hallucination in RAG systems.
        """,
        source="rag.pdf"

    )

    assistant.add_text(
        """
An AI agent is a system in which a large language model does not simply produce a single response, but operates in a loop: it observes the current state, decides on an action, executes that action through a tool, observes the result, and repeats until the task is complete. The key difference between a chatbot and an agent is autonomy over control flow. In a standard pipeline, the developer hard-codes each step; in an agent, the model dynamically decides which step to take next based on intermediate results.
 
The foundational pattern for agents is ReAct, short for Reasoning and Acting. In a ReAct loop, the model alternates between producing a reasoning step, which explains what it needs to do, and an action step, which calls a tool with specific arguments. The tool's output is appended to the conversation as an observation, and the model reasons again with this new information. Modern LLM APIs formalize this through native function calling, sometimes called tool use. The developer defines tools with a name, a natural-language description, and a JSON schema for the input parameters. The model then returns a structured tool call instead of free text, which the application executes and feeds back. Tool descriptions matter a great deal: vague or overlapping descriptions lead to wrong tool selection, while clear, distinct descriptions with explicit usage conditions make the agent far more reliable.
 
As tasks become more complex, a simple loop is often not enough, and developers model agents as explicit state machines or graphs. Frameworks such as LangGraph represent an agent as a directed graph where nodes are functions, such as calling the LLM, executing a tool, or validating output, and edges define transitions between them. Conditional edges route execution based on the current state, for example sending the flow back to the model if a tool returned an error, or forward to a response node once the answer is ready. The shared state object carries messages, intermediate results, and flags across the graph. Checkpointing persists this state after each step, which enables long-running workflows, recovery after failure, and human-in-the-loop patterns in which execution pauses for approval before a sensitive action such as sending an email or writing to a database.
 
Memory is another core component of agent design. Short-term memory is the conversation and scratchpad within the current run, bounded by the model's context window. When conversations grow long, strategies such as summarizing older turns or trimming tool outputs keep the context manageable. Long-term memory stores facts, user preferences, or past task results in an external store, often a vector database or key-value store, and retrieves them when relevant. Designing what to remember and when to recall it is as important as the retrieval mechanism itself.
 
Multi-agent architectures extend these ideas by splitting work across specialized agents. In a supervisor pattern, a coordinating agent routes subtasks to worker agents, each with its own tools and instructions, such as a research agent, a SQL agent, and a writing agent. This improves modularity and lets each agent use a focused prompt, but it also increases latency, cost, and the number of places where errors can occur. In practice, many production teams start with a single well-designed agent with a small set of reliable tools and only introduce multiple agents when a clear separation of responsibilities justifies the added complexity. Guardrails, including input validation, output schema checks, iteration limits, and timeouts, are essential to prevent agents from looping indefinitely or taking unintended actions.
        """,
        source="AI_agent.pdf"
    )

    assistant.add_text(
        """
        Agentic RAG combines the retrieval capabilities of RAG with the decision-making loop of an AI agent. In a classic RAG pipeline, every query follows the same fixed path: embed the question, retrieve the top-k chunks, and generate an answer. This works well for simple factual lookups, but it struggles with ambiguous questions, multi-hop reasoning, comparisons across documents, and cases where the first retrieval returns irrelevant context. Agentic RAG treats retrieval as a tool that the model can choose to call, call multiple times, reformulate, or skip entirely.
 
A typical agentic RAG workflow begins with query analysis and routing. The agent first classifies the incoming question: it might be small talk that needs no retrieval, a question for a specific knowledge base such as HR policies or technical manuals, a request that requires structured data from a SQL database, or a query that needs web search. Routing avoids unnecessary retrieval calls and sends each query to the most appropriate source. For complex questions, the agent may perform query decomposition, breaking a question like "compare the annual leave policy for full-time and contract staff" into separate sub-queries, retrieving evidence for each, and then synthesizing a combined answer.
 
Query rewriting is another important technique. User questions are often short, conversational, or dependent on earlier turns, for example "what about for managers?" A rewriting step uses the conversation history to transform this into a standalone, retrieval-friendly query. Some systems also use HyDE, Hypothetical Document Embeddings, where the model first generates a hypothetical answer and embeds that text, because a hypothetical answer is often closer in vector space to real answer passages than the raw question is.
 
The most distinctive feature of agentic RAG is self-correction. Patterns such as Corrective RAG add a grading step after retrieval, in which the model or a lightweight classifier evaluates whether each retrieved chunk is actually relevant to the question. If relevance is low, the agent rewrites the query and retrieves again, or falls back to another source. After generation, a second check can verify groundedness, confirming that every claim in the answer is supported by the retrieved context. If the answer contains unsupported statements, the agent regenerates or explicitly tells the user that the information was not found. In a graph framework like LangGraph, these checks are implemented as nodes with conditional edges, and a maximum retry count prevents infinite loops.
 
Evaluation and observability are what separate a demo from a production system. RAG evaluation typically measures retrieval and generation separately. Retrieval metrics include recall at k, precision at k, and Mean Reciprocal Rank, calculated against a labeled set of questions with known relevant chunks. Generation metrics include faithfulness, which measures whether the answer is supported by the context; answer relevance, which measures whether it actually addresses the question; and context precision, which measures how much of the retrieved context was useful. Frameworks such as RAGAS and DeepEval automate many of these metrics using LLM-as-a-judge scoring. On the observability side, tracing tools such as Langfuse or LangSmith record every step of an agent run, including prompts, retrieved chunks, tool calls, token usage, latency, and cost. These traces are essential for debugging why an agent chose a wrong tool, why retrieval missed a relevant document, or why latency spiked. Combining a curated evaluation dataset, automated regression tests, and production tracing creates a feedback loop that allows teams to improve chunking, retrieval, prompts, and agent logic with measurable confidence.
        """,
        source="agentic_rag.txt"
    )

    print(f"\nTotal chunks indexed: {assistant.get_document_count()}")
    print(f"Sources: {assistant.list_sources()}")


    # Show basic retriever return for a vague query
    # retriever = assistant._build_retriever()
    # docs = retriever.invoke("What tools help me build AI apps?")

    # print("BASIC RETRIEVER RESULTS:\n")
    # for i, doc in enumerate(docs):
    #     source = doc.metadata.get("source", "Unknown")
    #     print(f"    Chunk {i+1} [{source}]: {doc.page_content[:100]}...")
    #     print()

    # For debugging
    import logging
    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.DEBUG)

    retriever = assistant._build_retriever(use_advanced=True)
    docs = retriever.invoke("What tools help me build AI apps?") # uses LLM to create multiple similar queries for better results (create same question but in different way of asking)

    print(f"\nMulti-query returned {len(docs)} unique chunks")

    logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.WARNING)


    session = "demo"






    # --- Question 1: Direct answer ---
    print("\n" + "=" * 60)
    print("QUESTION 1: Direct factual question")
    print("=" * 60)

    q1 = "What is RAG and what are the main components?"
    print(f"\nUser: {q1}")
    print(f"\nAssistant: {assistant.ask(q1, session)}")

    # --- Question 2 ---
    print("\n" + "=" * 60)
    print("QUESTION 2: Requires info from multiple sources")
    print("="*60)

    q2 = "How does the Agentic RAG works?"
    print(f"\nUser: {q2}")
    print(f"\nAssistant: {assistant.ask(q2, session)}")

    # --- Question 3
    print("\n" + "=" * 60)
    print("QUESTION 3: Follow-up (this will fail)")
    print("="*60)

    q3 = "Can you expand ont the second component you just mentioned?"
    print(f"\nUser: {q3}")
    print(f"\nAssistant: {assistant.ask(q3, session)}")

    # It will fail because there is no memory --> after adding memory it will work
    print("\n" + "=" * 60)
    print("PROBLEM: It has no idea what 'you just mentioned' means")
    print("Each question is independent -- there is no memory")
    print("We fix this in the next video.")
    print("="*60)


    # --- Show history ---
    print("\n"+ "="*60)
    print("CONVERSATION HISTORY (proof it's tracked)")
    print("="*60)

    for i, msg in enumerate(assistant.get_session_history_display(session)):
        role = "USER" if msg["role"] == "human" else "AI"
        content = (
            msg["content"][:120] + "..."
            if len(msg["content"]) > 120

            else msg["content"]
        )

        print(f"\n {i+1}. [{role}]: {content}")

    # Prove sessions are isolated
    print("\n" + "=" *60)
    print("SESSION ISOLATION")
    print("="*60)

    q4 = "What did we discuss so far"

    print(f"\nUser (session == 'demo'):     {assistant.ask(q4, 'demo')[:150]}")
    print(f"\nUser (session == 'new'):     {assistant.ask(q4, 'new')[:150]}") # Each session is own isolated instance
    print("\n'new' session has no idea -- different memory")

    # Bonus: Show the persist directory exists on disk
    import os

    print(f"\nFiles on disk: {os.listdir("./research_db")}")
    print("This data survices a restart!")



