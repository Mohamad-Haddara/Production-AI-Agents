# Basic Embeddings + Batch Embeddings + Similarity Search
from tempfile import TemporaryFile
from langchain_community.document_loaders import directory
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def basic_embeddings():


    # single text
    text = "What is Machine Learning?"
    single_embedding = embeddings.embed_query(text)

    print(f"Vector Dimensions: {len(single_embedding)}")
    print(f"First 5 values: {single_embedding[:5]}")
    print(f"Vector norm: {np.linalg.norm(single_embedding):.4f}")



def batch_embeddings():
    text = [
        "What is Machine Learning?",
        "Explain the concept of overfitting in ML",
        "How does a neural network work?"
    ]


    batch_embeddings = embeddings.embed_documents(text)


    for i, emb in enumerate(batch_embeddings):
        print(f"Text {i+1} - Vector Dimensions: {len(emb)}")
        print(f"Text {i+1} - First 5 values: {emb[:5]}")
        print(f"Text {i+1} - Vector norm: {np.linalg.norm(emb)}")

# Understand for indexing rag system
def similarity_search():
    #  Documents
    docs = []
    query = "Wjat programming language exist?"


    #embed documents and query
    doc_vector = embeddings.embed_documents(docs)
    query_vector = embeddings.embed_query(query)


    # compute cosine similarity
    def cosine_similarity(vec1, vec2):
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    similarities = [cosine_similarity(query_vector, doc_vec) for doc_vec in docs]

    # rank document by similarity
    ranked_docs = sorted(zip(docs, similarities), key = lambda x: x[1], reverse=True)

    print(f"Query: {query}\n")
    print("Ranked by similarity:")
    for doc, score in ranked_docs:
        print(f"    {score:.4f}: {doc}")



# Embedding Caching (V.I) - bcz avoid redundant API calls, because most cases we call embedding model and inferring it- To avoid redundant calls, we use caching
def embedding_caching():
    from langchain_classic.embeddings.cache import CacheBackedEmbeddings
    from langchain_classic.storage import LocalFileStore
    import tempfile

    with tempfile.TemporaryDirectory() as tempdir:
        store = LocalFileStore(directory = tempdir)

        cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings = embeddings,
            document_embedding_cache = store,
            namespace = "excercise"
    )

    text = "What is the Reinforcement Learning?"

    # First call - hits API - then saved in cache ~ for second call
    print("First call (API)")
    vectors1 = cached_embeddings.embed_documents([text])
    print(f"    Embedded {len(vectors1)} documents")

    # Second call - from cache
    print("\nSecond call (Cache):")
    vectors2 = cached_embeddings.embed_documents([text])
    print(f"    Embedded {len(vectors2)} documents")

    # Verify same results
    print(f"\nSame vectors: {np.allclose(vectors1[0], vectors2[0])}")

if __name__ == "__main__":
    #basic_embeddings()
    #batch_embeddings()
    embedding_caching()