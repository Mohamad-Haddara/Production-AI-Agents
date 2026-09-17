from dotenv.parser import Original
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
from dotenv import load_dotenv


load_dotenv()


embedding_models = OpenAIEmbeddings(model="text-embedding-3-small")


sample_docs = [
    Document(
        page_content="Employees accrue 2.5 days of annual leave per completed month of service. "
                     "Unused leave may be carried over to the following year up to a maximum of 15 days; "
                     "any excess is forfeited on 31 December.",
        metadata={"source": "hr_policy_2025.pdf", "topic": "leave"},
    ),
    Document(
        page_content="Expense claims must be submitted within 30 days of the transaction date. "
                     "Claims above QAR 5,000 require line-manager approval and an original tax invoice. "
                     "Reimbursement is processed in the payroll cycle following approval.",
        metadata={"source": "finance_handbook.pdf", "topic": "expenses"},
    ),
    Document(
        page_content="All production data must remain within the Qatar Central region. "
                     "Cross-border transfer of personal data requires a documented lawful basis and "
                     "written approval from the Data Protection Officer.",
        metadata={"source": "data_governance.md", "topic": "compliance"},
    ),
    Document(
        page_content="The retrieval service exposes POST /v1/search. It accepts a query string, an "
                     "optional tenant_id filter, and top_k (default 5). Responses include document "
                     "chunks with similarity scores and source metadata.",
        metadata={"source": "api_reference.md", "topic": "engineering"},
    ),
    Document(
        page_content="New joiners complete IT onboarding on day one: account provisioning, MFA enrolment, "
                     "and a security awareness briefing. Laptop handover requires a signed asset form.",
        metadata={"source": "onboarding_guide.docx", "topic": "onboarding"},
    ),
    Document(
        page_content="Remote work is permitted up to two days per week with manager approval. "
                     "Employees must be reachable during core hours of 09:00–15:00 and connect only "
                     "through the corporate VPN.",
        metadata={"source": "hr_policy_2025.pdf", "topic": "remote_work"},
    ),
]



def chroma_basics():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
        # create vector store from documents - pass documents and embedding models
        vectorstore = Chroma.from_documents(
            documents=sample_docs,
            embedding=embedding_models,
            persist_directory= tempdir
        )

        # how many collections are created
        print(f"Vector store created {vectorstore._collection.count()} amd persisted")


        # perform similarity search
        query = "According to HR policy, how many core hours ?"
        results = vectorstore.similarity_search(query, k=2) # return 2 relevant documents

        print(f"Top 2 results for query {query}")
        for i, doc in enumerate(results):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")


def similarity_search_with_stores():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
        # create a vector from documents
        vectorstore = Chroma.from_documents(
            documents=sample_docs,
            embedding= embedding_models,
            persist_directory=tempdir,
        )


        # perform similarity search with score
        query = "According to HR policy, how many core hours?"
        results_with_scores = vectorstore.similarity_search_with_score(query=query, k=3)

        print(f"Top 3 results with scores for query `{query}`:")
        for idx, (doc, score) in enumerate(results_with_scores):
            print(f"Results {idx+1}: {doc.page_content} (Score: {score:.4f}, Source: {doc.metadata['source']}) ")


# Metadata filtering
def metadata_filtering():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
        # create a vector store from documents
        vectorstore = Chroma.from_documents(
            documents = sample_docs,
            embedding = embedding_models,
            persist_directory = tempdir
        )

        # perform similarity search with metadata filtering
        query = "According to HR policy, how many core hours?"
        


        # without metadata filtering
        results = vectorstore.similarity_search(query, k=3)

        print(f"Result without metadata filtering for query: {query}")
        for idx, doc in enumerate(results):
            print(f"Results {idx+1}: {doc.page_content} (Source: {doc.metadata['source']})")


        # With metadata filtering
        filter_criteria = {"topic": "remote_work"} # filtering criteria is dict


        print(f"Result with metadata filtering for query {query}")
        filtered_result = vectorstore.similarity_search(
            query=query,
            k=3,
            filter=filter_criteria # take filtering into consideration
        )

        for idx, doc in enumerate(filtered_result):
            print(f"Result {idx + 1}: {doc.page_content} (Source: {doc.metadata['source']})")




# Persist - save locally our  chroma db
def persist_chroma():
    # Set a directory
    persist_dir = "./chroma_db/"

    vectorstore = Chroma.from_documents(
        documents = sample_docs,
        embedding = embedding_models,
        persist_directory = persist_dir,
    )

    original_count = vectorstore._collection.count()

    print(f"Persisted vector store with: {original_count} documents.")
    print(f"Vector store persisted at: {persist_dir}")

    # simulator restart - load from distk
    del vectorstore

    reloaded = Chroma(
        embedding_function= embedding_models,
        persist_directory = persist_dir,
    )

    reloaded_count = reloaded._collection.count()
    print(f"Reloaded vector store with {reloaded_count} documents.")


    # verify search still works

    result = reloaded.similarity_search("timing hours", k=2)


# Vector store as a Retriever for chains (v.I)
def as_retriever():
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tempdir:
        vectorstore = Chroma.from_documents(
            documents = sample_docs,
            embedding = embedding_models,
            persist_directory=tempdir 
        )


        # basic retriever usage - allow us to call invoke function on it, bcz it is a chain (runnable)
        retriever = vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":3} # return 3 documents
        )


        # use retriever to get relevant documents
        docs = retriever.invoke("What are the working hours for employee?")

        print("Retriever results:")
        for i, doc in enumerate(docs):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")


        # Second type of retriever
        # MMR ~ Maximum Marginal Relevance -> give us diverse result

        mmr_retriever = vectorstore.as_retriever(
            search_type = "mmr",
            search_kwargs = {"k":3, "fetch_k": 5} # fetch 5 docs and return 3 diverse
        )


        mmr_docs = mmr_retriever.invoke("What are the working hours for employee?")
        print("\nMMR Retriever results:")
        for i, doc in enumerate(mmr_docs):
            print(f"Result {i+1}: {doc.page_content} (Source: {doc.metadata['source']})")


def exercise_vector_store_setup():
    """
    1. Takes a list of text strings
    2. splits them into chunks
    3. Stores in Chroma
    4. Returns a configured retriever

    Test with sample documents
    """

    def create_retriever(
        text: list[str],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        k: int =3
    ):

        # create a document
        docs = [Document(page_content = t) for t in text]

        # Split document
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap
        )

        chunks = splitter.split_documents(docs)

        #Create a vector database (in-memory for exercise)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding = embedding_models,
         
        )

        # Set a query
        query = ""

        # retrieve top 3 relevant chunks
        vectorstore.as_retriever(
            search_type = "similarity",
            search_kwargs = {"k":k}
        )


if __name__ == "__main__":

    #chroma_basics()
    #similarity_search_with_stores()
    #metadata_filtering()
    #persist_chroma()
    as_retriever()