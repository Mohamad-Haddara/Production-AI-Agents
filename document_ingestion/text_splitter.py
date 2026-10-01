"""
Text Splitters and Chunking Strategies
Optimizing document chunks for RAG

"""

from re import split
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    TokenTextSplitter,
    MarkdownHeaderTextSplitter,
    Language
)


from langchain_core.documents import Document
from dotenv import load_dotenv


load_dotenv()

sample_text = """
The dawn broke quietly over the coastal town of Oakhaven, casting a pale gold light across 
the dew-covered rooftops and cobblestone alleys. Early morning mist drifted off the bay, 
wrapping the sleeping fishing vessels in a soft, woolen haze. Down at the harbor, the 
rhythmic clinking of rigging against aluminum masts provided a steady background beat 
to the gentle, repeating slap of tide against timber pilings. 

Arthur adjusted his heavy wool coat, taking a slow sip from his steaming mug of black coffee. 
For forty years, he had watched the sun rise over this same stretch of ocean, yet the subtle 
shift in color from deep indigo to warm amber never failed to hold his attention. Every morning 
brought a slightly different rhythm—some days carried the crisp, decisive breeze of clear 
weather, while others portended the slow, heavy approach of offshore rain.

Key Morning Observations:
- Harbor Atmosphere: Thick coastal mist and rhythmic clinking of boat rigging.
- Weather Patterns: Quick shifts from deep indigo to warm amber hues on the water.
- Daily Routine: The local bakery opens as streetlights power down across the town.

Further up the hill, the town was beginning to awaken. A faint trail of smoke spiraled upward 
from the chimney of the local bakery, carrying the faint, comforting aroma of toasted grain 
and melting butter down the street. The first streetlights hummed quietly as they flicked off, 
yielding to the increasing daylight. It was a simple, predictable sequence of events, but to 
those who lived here, it was the steady foundation upon which everything else was built.
"""


sample_code = """
def process_environmental_data(records: list[dict]) -> dict:
    \"\"\"Filters and aggregates sensor readings from coastal monitoring stations.\"\"\"
    processed_summary = {
        "valid_readings": 0,
        "average_temperature": 0.0,
        "high_humidity_alerts": []
    }
    
    total_temp = 0.0
    
    for record in records:
        # Validate data integrity
        if not record.get("sensor_id") or record.get("status") != "ACTIVE":
            continue
            
        temp = record.get("temperature_c", 0.0)
        humidity = record.get("humidity_pct", 0.0)
        
        total_temp += temp
        processed_summary["valid_readings"] += 1
        
        # Check thresholds
        if humidity > 85.0:
            processed_summary["high_humidity_alerts"].append({
                "station": record["sensor_id"],
                "humidity": humidity
            })
            
    if processed_summary["valid_readings"] > 0:
        processed_summary["average_temperature"] = round(
            total_temp / processed_summary["valid_readings"], 2
        )
        
    return processed_summary

# Sample payload execution
if __name__ == "__main__":
    data_stream = [
        {"sensor_id": "BAY_01", "temperature_c": 14.2, "humidity_pct": 88.5, "status": "ACTIVE"},
        {"sensor_id": "BAY_02", "temperature_c": 13.8, "humidity_pct": 79.0, "status": "ACTIVE"},
        {"sensor_id": "HILL_01", "temperature_c": 12.5, "humidity_pct": 91.2, "status": "MAINTENANCE"}
    ]
    print(process_environmental_data(data_stream))
"""


def demo_recursive_splitter():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_text(sample_text)

    
    print(f"Original length: {len(sample_text)} chars")
    print(f"Number of chunks: {len(chunks)}")
    print(f"Data-Type of chunks: {type(chunks)}")
    print(f"Chunk sizes: {[len(c) for c in chunks]}")
    print(f"\nFirst chunk preview:\n{chunks[0][:200]}...")



# Chunk size comparison - is V.I - chunk size will determine whether we are saving the correct amount of context into vector DB (RAG system)
def chunk_size_comparison():
    sizes = [200, 500, 1000]

    print("=== Chunk Size Comparison ===")
    for size in sizes:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = size,
            chunk_overlap = size // 5 # 20% overlap 
        )

        chunks = splitter.split_text(sample_text)
        print(f"    Size {size}: {len(chunks)} chunks")

#Overlap is very important - difference between finding an answer and finding a complete answer.

def markdown_splitter():
    headers_to_consider = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3")
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_consider
    )

    chunks = splitter.split_text(sample_text)

    print(f"Markdown Splitter produced {len(chunks)} chunks.")
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i+1} --- \n")
        print(f"Metadata: {chunk.metadata}\n")
        print(f"Content: {chunk.page_content[:200]}\n")

# Code splitter technique
def code_splitter():
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON, # Very important, so splitter can understand the python syntax
        chunk_size = 500,
        chunk_overlap = 50
    )

    chunks = python_splitter.split_text(sample_code)

    print(f"\nCode Splitter produced {len(chunks)} chunks")
    for i, chunk in enumerate(chunks):
        print(f"Chunk {i} ({len(chunk)} chars):")
        print(chunk[:150] + "..." if len(chunk) > 150 else chunk)


# PDF Document Splitting
def document_splitter():
    from langchain_community.document_loaders import PyPDFLoader
    from langchain_core.documents import Document 

    loader = PyPDFLoader("C:/Users/mhaddara/OneDrive - Education Above All/Documents/AI Appliction/Production AI Agents/docs/Cheat-Sheet-Kubernetes.pdf")
    docs = loader.load() # Return list of document object (1 per page)

    print(f"Loaded {len(docs)} documents from PDF.")


    splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50
    )

    # Split documents into chunks
    split_docs = splitter.split_documents(docs)

    print(f"Split into {len(split_docs)} chunks")
    print(f"\nFirst chunk metadata: {split_docs[0].metadata}")
    print(f"First chunk content: {split_docs[0].page_content[:200]}...")
    print(f"\nLast chunk metadata: {split_docs[-1].metadata}")

if __name__ == "__main__":
    print("=== Recursive Text Splitter ===")
    #demo_recursive_splitter()
    #chunk_size_comparison()

    #print("=== Markdown Header Text Splitter ===")
    #markdown_splitter()

    #print("=== Code Splitter===")
    #code_splitter()
    document_splitter()