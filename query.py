import os
import sys
from langchain.chains import RetrievalQA
#from langchain_community.vectorstores import Qdrant
#from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.llms import Ollama
from langchain_ollama import OllamaLLM
from langchain_ollama.embeddings import OllamaEmbeddings
#from langchain_qdrant import Qdrant
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
#from qdrant_client.http import models

#from langchain import hub
#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.runnables import RunnablePassthrough
#prompt = hub.pull("rlm/rag-prompt")

# Define constants
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")  # Default: Local Qdrant instance
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTION_NAME = "court-cases"

# Initialize Qdrant Client
try:
    client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)

    # Ensure the collection exists
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        print(f"Error: Qdrant collection '{COLLECTION_NAME}' not found. Please create it first.")
        sys.exit(1)

    print(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    print(f"Error initializing Qdrant: {e}")
    sys.exit(1)

# Initialize Ollama LLM (Local model)
llm = OllamaLLM(model="llama3.2")

# Initialize HuggingFace Embeddings for Vector Storage
#embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
embedding_model = OllamaEmbeddings(model="nomic-embed-text")


# Initialize Qdrant Vector Store
vector_store = QdrantVectorStore(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
)

# Display index statistics
try:
    collection_info = client.get_collection(COLLECTION_NAME)
    print(f"Collection Info: {collection_info}")
except Exception as e:
    print(f"Error fetching collection info: {e}")
    sys.exit(1)

# Set up Retriever and QA pipeline
retriever = vector_store.as_retriever(search_kwargs={"k": 6})
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever
)

def graceful_shutdown():
    """Handles script shutdown gracefully."""
    print("\nShutting down ... Goodbye!")
    sys.exit(0)

def process_query(query):
    """
    Processes a query by retrieving relevant documents from Qdrant 
    and using Ollama to generate a response.
    """
    try:
        # Retrieve documents
        #retrieved_docs = retriever.get_relevant_documents(query)
        retrieved_docs = retriever.invoke(query)

        if not retrieved_docs:
            print("\nNo relevant documents found.")
            return

        print(f"\nRetrieved {len(retrieved_docs)} documents. Displaying top document:\n")
        print(retrieved_docs[0])
        #print(retrieved_docs[0].page_content[:500])  # Show first 500 characters

        def format_docs(docs):
            return "\n\n".join(doc for doc in docs)

        # Query LLM
        #print(query)
        response = qa.invoke(query)

#        qa_chain = (
#            {
#                "context": vector_store.as_retriever() | format_docs,
#                "question": RunnablePassthrough(),
#            }
#            | prompt
#            | llm
#            | StrOutputParser()
#        )
#
#        qa_chain.invoke(query)

        # Ensure response structure is valid
        if isinstance(response, dict) and "result" in response:
            print(f"\nResponse: {response['result']}")
        else:
            print("\nUnexpected response format:", response)
#        print(qa_chain)
    except Exception as e:
        print(f"\nError processing query: {e}")

def query_loop():
    """Main loop for interactive querying."""
    print("Enter your query for court cases (type 'exit' to quit):")
    while True:
        try:
            query = input("> ").strip()

            # Exit condition
            if query.lower() in {"exit", "quit"}:
                graceful_shutdown()

            # Process the query
            if query:
                process_query(query)
            else:
                print("Please enter a valid query.")
        except KeyboardInterrupt:
            graceful_shutdown()

if __name__ == "__main__":
    query_loop()

