import os
import pickle
from langchain_community.retrievers import BM25Retriever
from config import PDF_FILE_PATH, LOCAL_STORE_PATH
from data_processing import SessionDocStore, process_and_ingest

def run_ingestion():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Enter your OPENAI_API_KEY for ingestion: ")
        
    if not os.path.exists(PDF_FILE_PATH):
        print(f"❌ PDF file not found at {PDF_FILE_PATH}")
        print("Please place a PDF file named 'document.pdf' in the pdf_mcp_server directory.")
        return

    print("🚀 Starting PDF Ingestion...")
    doc_store = SessionDocStore()
    
    # 1. Process and Ingest into Pinecone
    try:
        documents = process_and_ingest(PDF_FILE_PATH, doc_store, api_key)
    except Exception as e:
        print(f"❌ Failed during PDF processing: {e}")
        return

    # 2. Setup BM25 Retriever
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = 5

    # 3. Save local stores to disk
    local_data = {
        "doc_store": doc_store.store,
        "bm25_docs": documents  # We save the documents to rebuild BM25 on load
    }
    
    with open(LOCAL_STORE_PATH, "wb") as f:
        pickle.dump(local_data, f)
        
    print(f"✅ Ingestion Complete! Saved local data to {LOCAL_STORE_PATH}")
    print(f"Processed {len(documents)} chunks.")

if __name__ == "__main__":
    run_ingestion()
