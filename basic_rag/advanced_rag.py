"""
Advanced RAG Patterns
Multi-query, self-query, compression, hybrid search
"""

from multiprocessing.spawn import import_main_path
from langchain_classic.retrievers.multi_query import MultiQueryRetriever # asks LLM to rewrite our question several ways - retrieve for each - then unions the results. (fixes vocabs)
from langchain_classic.retrievers import ContextualCompressionRetriever, ParentDocumentRetriever # Filter the retrived chunks before they reach context window
from langchain_classic.retrievers.document_compressors import LLMChainExtractor # It run LLM over each document(chunk) to pull out only the relevant sentences. (reduce Cost bcz fewer tokens at scale and less noise and faster processing)
from langchain_classic.retrievers import EnsembleRetriever # Runs several retrievers and fuses their rankings with Reciprocal Rank Fusion, using weights I set. That's how I do hybrid search
from langchain_community.retrievers import BM25Retriever # Sparse keyword retrieval. Catches exact terms. Note, it is in memory only, rebuilt each run
from langchain_classic.storage import InMemoryStore # Key-value docstore, it is paired with ParentDocument or MultiVector retrieval so I embed small chunks but return full parent.
from langchain_chroma import Chroma # local vector store - Fine for development
from langchain_core import vectorstores
from langchain_openai import ChatOpenAI, OpenAIEmbeddings 
from langchain_core.documents import Document # Core object: page_content + metadata dict. Metadata is used for filtering, so populate it at ingestion
from langchain_text_splitters import RecursiveCharacterTextSplitter # split text into chunks based on separator order (paragraph, then lines, then words) to avoid cutting mid-thought
from langchain_core.prompts import ChatPromptTemplate # builds role-structured prompts with variable slots
from langchain_core.output_parsers import StrOutputParser  # convert model output into plain string
from langchain_core.runnables import RunnablePassthrough # LCEL glue
from dotenv import load_dotenv # reads .env into environment variables so the API keys get picked up
import logging # Relevant here because MultiQueryRetriever logs its generated query variations at INFO level, which is the only way to see what it actually searched for.

load_dotenv()

# Enable logging to see multi-query generation
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


 
# Knowledge base for demos

 
TECH_DOCS = [
    Document(
        page_content=(
            "An embedding maps a piece of text to a fixed-length vector of floats so that "
            "semantically similar texts land close together in the vector space. Closeness is "
            "usually measured with cosine similarity, which compares the angle between two "
            "vectors and ignores their magnitude. Because the comparison happens in vector "
            "space rather than on the literal characters, a query can match a passage that "
            "shares no words with it at all."
        ),
        metadata={"topic": "vector-search", "language": "python", "difficulty": "beginner", "doc_id": "vs-001"},
    ),
    Document(
        page_content=(
            "Hybrid retrieval runs a sparse keyword retriever and a dense vector retriever over "
            "the same corpus and fuses the two ranked lists. Reciprocal Rank Fusion scores each "
            "document as the sum of 1/(k + rank) across the lists, with k commonly set to 60, so "
            "a document ranked well by either retriever survives. This recovers exact matches on "
            "acronyms and product codes that pure vector search tends to drop."
        ),
        metadata={"topic": "vector-search", "language": "python", "difficulty": "intermediate", "doc_id": "vs-003"},
    ),
    Document(
        page_content=(
            "HNSW builds a layered proximity graph for approximate nearest-neighbour search. "
            "The M parameter sets how many bidirectional links each node keeps, and "
            "ef_construction controls how wide the candidate list is while the index is being "
            "built. Raising either improves recall at the cost of index build time and memory. "
            "At query time ef_search trades latency against recall and can be tuned per request "
            "without rebuilding the index."
        ),
        metadata={"topic": "vector-search", "language": "python", "difficulty": "advanced", "doc_id": "vs-002"},
    ),
    Document(
        page_content=(
            "A single blocking call stalls the entire event loop and every coroutine waiting on "
            "it. CPU-bound work, synchronous database drivers and time.sleep all have this "
            "effect. Push such calls onto a thread with asyncio.to_thread or "
            "loop.run_in_executor, or move them to a separate process for genuinely CPU-bound "
            "work. asyncio debug mode will log any callback that occupies the loop for longer "
            "than 100 milliseconds."
        ),
        metadata={"topic": "async", "language": "python", "difficulty": "advanced", "doc_id": "as-002"},
    ),
    Document(
        page_content=(
            "Tests that assert on wording rather than behaviour break every time a message is "
            "reworded, and tests that reach into private attributes break on every refactor. "
            "Both produce a suite people learn to ignore. Assert on the contract the caller "
            "depends on: the status code, the shape of the payload, the side effect that was "
            "or was not performed."
        ),
        metadata={"topic": "testing", "language": "python", "difficulty": "intermediate", "doc_id": "te-003"},
    ),
    Document(
        page_content=(
            "Opening a fresh connection per request is expensive under load, and Postgres "
            "allocates a backend process per connection, so unbounded clients exhaust the "
            "server. SQLAlchemy's pool_size and max_overflow cap what a single worker holds, "
            "while pool_pre_ping discards connections silently killed by a proxy or firewall. "
            "Across many workers an external pooler such as PgBouncer in transaction mode is the "
            "usual answer."
        ),
        metadata={"topic": "database", "language": "python", "difficulty": "advanced", "doc_id": "db-002"},
    ),
    Document(
        page_content=(
            "RFC 7807 defines a problem details object with type, title, status, detail and "
            "instance members, served as application/problem+json. Returning a consistent error "
            "shape lets clients branch on a stable type URI instead of pattern-matching prose. "
            "Rate-limited responses should carry HTTP 429 together with a Retry-After header so "
            "callers know how long to wait."
        ),
        metadata={"topic": "api-design", "language": "python", "difficulty": "intermediate", "doc_id": "ap-003"},
    ),
    Document(
        page_content=(
            "Each instruction in a Dockerfile produces a layer, and a layer is rebuilt only when "
            "it or anything above it changes. Copying the whole source tree before installing "
            "dependencies therefore invalidates the dependency layer on every code edit. Copy "
            "the lock file, install, then copy the source, and dependency installation is cached "
            "across ordinary changes."
        ),
        metadata={"topic": "docker", "language": "bash", "difficulty": "beginner", "doc_id": "dk-001"},
    ),
]
 

# Create base of vectorstore
def create_base_vectorstore():
    """Create a basic vector store for demos."""
    return Chroma.from_documents(
        documents=TECH_DOCS,
        embedding=OpenAIEmbeddings(model="text-embedding-3-small")
    ) # Return VectorStore Object


def multi_query_retriever():
    """
    Multi-Query Retriever generates multiple query.
    """
    print("="*60)
    print("Multi-Query Retriever")
    print("Generate multiple query on our question ~ rewrite question in many ways")
    print("="*60)


    vectorstore = create_base_vectorstore()
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


    # Create multi-query retriever -> Form one query will llm to generate multiple queries ==> better results
    retriever = MultiQueryRetriever.from_llm(
        retriever = vectorstore.as_retriever(search_kwargs = {"k":3}), llm = llm
    )


    query = "What is HNSW and how ot works?"

    print(f"\nOriginal Query: {query}")
    print("\nThe retriever will generate multiple query variation...")
    

    # Retrieve documents ~ using multi-query retriever
    docs = retriever.invoke(query)


    print(f"Retrieved {len(docs)} unqiue documents:")

    for i, doc in enumerate(docs):
        print(
            f"\n{i+1}. [{doc.metadata.get('topic','N/A')}] {doc.page_content[:100]}"
        )







# Contextual Compression
def contextual_compression():
    """
    Contextual Compression extracts only relevant parts.
    
    lead to extra call for LLM for completion but worth for large document with mixed content, and we need a precise information.
    Token cost matter at scale.

    """

    print("="*60)
    print("CONTEXTUAL COMPRESSION RETRIEVER")
    print("Extract only query-relevant content from documents")
    print("="*60)

    vector_store = create_base_vectorstore()

    llm =  ChatOpenAI(model="gpt-4o-mini", temperature=0.3)


    # Create a compressor - use llm to extract only relevant part - LLM read each chunk
    compressor = LLMChainExtractor.from_llm(llm)

    # Wrap retriever with compression
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=vector_store.as_retriever(search_kwargs={"k":4})
    )



    query = "What is HNSW ?"

    print(f"\nQuery: {query}")
    
    # without compression
    base_docs = vector_store.as_retriever(search_kwargs = {"k":2}).invoke(query)
    print(f"\n--- WITHOUT Compression (full chunks) ---")
    for doc in base_docs:
        print(f"length: {len(doc.page_content)} chars")
        print(f"Content: {doc.page_content[:150]}...\n")

    

    # With compression
    compressed_docs = compression_retriever.invoke(query)
    print(f"\n--- WITH Compression (relevant only) ---")
    for doc in compressed_docs:
        print(f"Length: {len(doc.page_content)}")
        print(f"Content: {doc.page_content}\n")



# Hybrid Search
def hybrid_search():
    """
   Hybrid search combining keyword (BM25) and semantic search. 
    """

    print("="*60)
    print("ENSEMBLE/HYBRID RETRIEVER")
    print("Combines keyword (BM25) + semantic search")
    print("="*60)


    vector_store = create_base_vectorstore()



    # BM25 keyword retriever 
    bm25_retriever = BM25Retriever.from_documents(TECH_DOCS)
    bm25_retriever.k = 3

    
    # Semantic retriever
    semantic_retriever = vector_store.as_retriever(search_kwargs = {"k":3})

    # Combine both retrievers
    ensemble_retriever = EnsembleRetriever(
        retrievers = [bm25_retriever, semantic_retriever],
        weights=[0.4, 0.6] # 40% keyword, 60% semantic - select % depends on use cases
    )


    queries = [
        "What is the difference between async and sync",
        "What is hybrid search and how it runs?"
    ]


    for query in queries:
        print(f"f\nQuery:{query}")
        print("-"*40)


        # compare results
        bm25_results = bm25_retriever.invoke(query)
        semantic_results = semantic_retriever.invoke(query)
        ensemble_results = ensemble_retriever.invoke(query)


        print(f"BM25 top result: {bm25_results[0].page_content[:60]}...")
        print(f"semantic top result: {semantic_results[0].page_content[:60]}...")
        print(f"Ensemble top result: {ensemble_results[0].page_content[:60]}...")



# Parent Document Retriever
def parent_document_retriever():
    """
    Parent Document Retriever: small chunks for search, large for context.

    because small chunk are connected to their parent which is large chunk
    """

    print("="*60)
    print("PARENT DOCUMENT RETRIEVER")
    print("Small chunks for precise search, large chunks for context")
    print("="*60)


    # Splitters
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size = 800, chunk_overlap = 100)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size = 200, chunk_overlap = 20)


    # Storage
    vector_store = Chroma(
        collection_name = "parent_child",
        embedding_function=OpenAIEmbeddings(model="text-embedding-3-small")
    )

    store = InMemoryStore()

    
    # Create a retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore = store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Add document
    retriever.add_documents(TECH_DOCS)

    # Search
    query = "what is HNSW?"

    print(f"\nQuery:{query}")

    # Regular retrieval (would get small chunks)
    child_docs = vector_store.similarity_search(query=query, k=1)
    print(f"\n --- Child Chunk (what search found) ---")
    print(f"Length: {len(child_docs[0].page_content)} chars")
    print(f"Content: {child_docs[0].page_content}")

    # Parent retrieval (get full context)
    parent_docs = retriever.invoke(query)
    print(f"\n--- Parent Chunk (what's returned) ---")
    print(f"Length: {len(parent_docs[0].page_content)} chars")
    print(f"Content: {parent_docs[0].page_content}")

    """
    Two stage process:
    Step 1: search for small chunks
    Step 2: We returned parent chunk

    For Parent is best for both (small chunk for high precision for search and large chunk is for complete context so LLM get right answer needed) ~ Parent doc has both of them and embedding are focused in PDOC retriever
    """

def advanced_rag_chain():
    """
    Complete RAG chain with advanced retrieval.

    Advanced RAG strategies to make system better.
    """
    print("="*60)
    print("COMPLETE ADVANCED RAG CHAIN")
    print("Multi-query + Compression + RAG")
    print("="*60)

    vectorstore = create_base_vectorstore()

    llm = ChatOpenAI(model = "gpt-4o-mini", temperature=0)


    # Implement multi-query for better recall ~ create multiple version of main query
    multi_retriever = MultiQueryRetriever.from_llm(
        retriever = vectorstore.as_retriever(search_kwargs = {"k":3}),
        llm=llm
    )


    # Compression to focus on relevant info ~ For precision 
    compressor = LLMChainExtractor.from_llm(llm)


    advanced_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_retriever
    )


    # RAG Prompt
    prompt = ChatPromptTemplate.from_template(
        """
        Answer the question based on the following context. Be specific and cite which tech

        Context:
        {context}

        Question: {question}
        
        Answer:
        """
    )


    def format_doc(docs):
        return "\n\n".join(doc.page_content for doc in docs)


    # Build chain
    rag_chain = (
        {"context": advanced_retriever | format_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    # Run RAG
    questions = [
        "What is HNSW?",
        "What is the difference between async and sync in programming language, and it is considered as advanced topic or intermediate?"
    ]

    for q in questions:
        print(f"\nQ:{q}")
        answer = rag_chain.invoke(q)
        print(f"Answer:{answer}")




if __name__ == "__main__":
    
    #multi_query_retriever()
    #contextual_compression()
    #hybrid_search()
    #parent_document_retriever()
    advanced_rag_chain()