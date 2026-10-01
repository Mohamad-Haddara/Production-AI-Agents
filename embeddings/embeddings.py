from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()



# embeddings = OpenAIEmbeddings(model="text-embedding-3-small") #1536 dimensions


# # single text embedding
# text = "This is a sample text to be embedded."
# embedding = embeddings.embed_query(text)
# #print(f"Embedding for single text: {embedding}")
# print(len(embedding))


# # Multi-text embedding
# embeds = embeddings.embed_documents([
#    "This is the first document",
#    "This is the second document" 
# ])

# print(f"Embeddings for multiple texts: {embeds}")
# print(f"Number of embeddings returned: {len(embeds)}") # 2
# print(f"Length of each embedding: {len(embeds[0])}") # 1536
# print(f"Type of embeds: {type(embeds)}")


# ===== Free Embedding Models using HuggingFace =====
embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2") # 384 dimensions
