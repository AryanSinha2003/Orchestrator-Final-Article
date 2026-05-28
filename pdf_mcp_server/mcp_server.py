import os
import pickle
from mcp.server.fastmcp import FastMCP
from langchain_community.retrievers import BM25Retriever
from config import LOCAL_STORE_PATH
from data_processing import SessionDocStore
from rag_engine import run_advanced_rag

# Initialize the MCP Server
mcp = FastMCP("PDF-QA-Server", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

# Global variables to hold the retrievers
_bm25_retriever = None
_doc_store = None

def load_local_stores():
    global _bm25_retriever, _doc_store
    
    if _bm25_retriever is not None and _doc_store is not None:
        return
    
    if not os.path.exists(LOCAL_STORE_PATH):
        raise FileNotFoundError(f"Local store {LOCAL_STORE_PATH} not found. Did you run ingest.py first?")
        
    print(f"Loading local stores from {LOCAL_STORE_PATH}...")
    with open(LOCAL_STORE_PATH, "rb") as f:
        local_data = pickle.load(f)
        
    _doc_store = SessionDocStore()
    _doc_store.store = local_data["doc_store"]
    
    documents = local_data["bm25_docs"]
    _bm25_retriever = BM25Retriever.from_documents(documents)
    _bm25_retriever.k = 5
    print("✅ Local stores loaded successfully.")

@mcp.tool()
def query_pdf(query: str, openai_api_key: str = "") -> str:
    """
    Query the ingested PDF document to extract information and answer questions.
    Uses Hybrid Search (BM25 + Pinecone) and GPT-5 to return a robust answer.
    """
    try:
        load_local_stores()
        
        if not openai_api_key:
            return "Error: openai_api_key must be provided to query the PDF."
            
        # Execute RAG pipeline
        answer = run_advanced_rag(query, _bm25_retriever, _doc_store, openai_api_key)
        return answer
        
    except Exception as e:
        return f"Error executing RAG pipeline: {str(e)}"

if __name__ == "__main__":
    # Preload the models on startup
    try:
        load_local_stores()
    except Exception as e:
        print(f"Warning during startup: {e}")
        print("You must run ingest.py before the server can query the PDF.")
        
    # Render sets PORT, default to 8001
    mcp.run(transport='sse')