import os
import uuid
import json
import pickle
import concurrent.futures
from typing import List, Dict
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_core.messages import HumanMessage
from pinecone import Pinecone

from config import *

# Initialize Pinecone Client
if PINECONE_API_KEY:
    pc = Pinecone(api_key=PINECONE_API_KEY)
else:
    pc = None

class SessionDocStore:
    """
    In-memory storage for heavy content (Images/Tables) tied to a session/document.
    """
    def __init__(self):
        self.store = {}

    def save_chunk(self, doc_id: str, data: Dict):
        self.store[doc_id] = data

    def get_chunk(self, doc_id: str):
        return self.store.get(doc_id, {})
    
    def clear(self):
        self.store = {}

    def save_to_disk(self, file_path: str):
        with open(file_path, 'wb') as f:
            pickle.dump(self.store, f)
            
    def load_from_disk(self, file_path: str):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.store = pickle.load(f)

def create_multimodal_summary(text, tables, images, api_key: str):
    llm = get_llm(api_key=api_key)
    prompt_text = f"Analyze content. TEXT: {text[:1000]}. INSTRUCTIONS: Summarize text and describe tables in detail without missing out /skipping any information."
    
    message_content = [{"type": "text", "text": prompt_text}]
    if images:
        for b64_str in images:
            if "," in b64_str: b64_str = b64_str.split(",")[1]
            message_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_str}", "detail": "low"}
            })
            
    response = llm.invoke([HumanMessage(content=message_content)])
    return response.content

def process_single_chunk(i, chunk, doc_store, api_key: str):
    """
    Worker function to process a single chunk in a separate thread.
    """
    content = {'text': chunk.text, 'tables': [], 'images': []}
    
    # Extract visual data
    if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
        for element in chunk.metadata.orig_elements:
            el_type = type(element).__name__
            if el_type == 'Table':
                content['tables'].append(getattr(element.metadata, 'text_as_html', element.text))
            elif el_type == 'Image' and hasattr(element.metadata, 'image_base64'):
                content['images'].append(element.metadata.image_base64)

    # Deciding whether to call LLM (Slow) or just use text (Fast)
    if content['images'] or content['tables']:
        enhanced_text = create_multimodal_summary(content['text'], content['tables'], content['images'], api_key)
    else:
        enhanced_text = content['text']

    doc_id = str(uuid.uuid4())
    
    # Save heavy data to local store
    doc_store.save_chunk(doc_id, {
        "raw_text": content['text'],
        "tables": content['tables'],
        "images": content['images']
    })
    
    # Return the processed Document
    return Document(
        page_content=enhanced_text,
        metadata={"doc_id": doc_id, "chunk_index": i}
    )

def process_and_ingest(file_path: str, doc_store: SessionDocStore, api_key: str):
    if not pc:
        raise ValueError("PINECONE_API_KEY is not set.")
    
    print(f"📄 Partitioning: {file_path}")
    
    elements = partition_pdf(
        filename=file_path, 
        strategy="auto", 
        infer_table_structure=True,
        extract_image_block_types=["Table","Images"], 
        extract_image_block_to_payload=True
    )
    
    chunks = chunk_by_title(elements, max_characters=2000, new_after_n_chars=1500, combine_text_under_n_chars=300)
    
    documents = []
    print(f"🔄 Processing {len(chunks)} chunks in parallel...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_chunk = {
            executor.submit(process_single_chunk, i, chunk, doc_store, api_key): i 
            for i, chunk in enumerate(chunks)
        }
        
        futures_list = list(future_to_chunk.keys())
        
        for future in futures_list:
            try:
                doc = future.result()
                documents.append(doc)
            except Exception as e:
                print(f"❌ Error processing chunk: {e}")

    print(f"🔮 Ingesting {len(documents)} vectors to Pinecone Index: {INDEX_NAME}")
    
    # Clear index first if necessary, or just use a specific namespace. For this single-file app, we can use a hardcoded namespace
    namespace = "single-pdf-mcp"
    
    PineconeVectorStore.from_documents(
        documents=documents,
        index_name=INDEX_NAME,
        embedding=get_embeddings(),
        namespace=namespace 
    )
    
    return documents
