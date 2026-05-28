import os
# Prevent ONNX Runtime from hanging on restricted CPU environments (Render Free Tier)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import logging
from typing import List
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from flashrank import Ranker, RerankRequest
from config import *

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fast CPU Re-ranking
ranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2", cache_dir="/tmp/flashrank_cache")

def reciprocal_rank_fusion(results: List[List[Document]], k=60):
    fused_scores = {}
    doc_map = {}
    for docs in results:
        for rank, doc in enumerate(docs):
            doc_id = doc.metadata.get("doc_id")
            if doc_id not in doc_map: doc_map[doc_id] = doc
            if doc_id not in fused_scores: fused_scores[doc_id] = 0
            fused_scores[doc_id] += 1 / (rank + k)
    
    reranked_ids = sorted(fused_scores, key=fused_scores.get, reverse=True)
    return [doc_map[doc_id] for doc_id in reranked_ids]

def rerank_documents(query: str, docs: List[Document], top_n=5):
    """
    Optimized Re-ranking using FlashRank.
    """
    if not docs: return []

    try:
        # Prepare format for FlashRank
        passages = [
            {"id": str(i), "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(docs)
        ]
        
        # Rerank
        rerank_request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(rerank_request)
        
        # Convert back to Document objects
        final_docs = []
        for res in results[:top_n]:
            final_docs.append(Document(page_content=res["text"], metadata=res["meta"]))
            
        return final_docs
    except Exception as e:
        logger.error(f"FlashRank failed: {e}")
        return docs[:top_n] # Fallback

def run_advanced_rag(query: str, bm25_retriever, doc_store, openai_api_key: str):
    namespace = "single-pdf-mcp"
    queries = [query]
    
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=get_embeddings(), namespace=namespace
    )
    dense_retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    all_docs = []
    
    # Hybrid Search
    for q in queries:
        dense_docs = dense_retriever.invoke(f"query: {q}")
        sparse_docs = bm25_retriever.invoke(q)
        # Fuse results
        all_docs.extend(reciprocal_rank_fusion([dense_docs, sparse_docs]))
        
    # Deduplicate by ID
    unique_docs = {d.metadata["doc_id"]: d for d in all_docs}
    
    # Re-rank
    final_docs = rerank_documents(query, list(unique_docs.values()), top_n=5)
    
    # Context Construction
    context_text = ""
    retrieved_images = []
    seen_imgs = set()
    
    for i, doc in enumerate(final_docs):
        # Fetch heavy content from in-memory store
        heavy = doc_store.get_chunk(doc.metadata["doc_id"])
        
        context_text += f"\n--- Source {i+1} ---\n{heavy.get('raw_text', '')}\n"
        for t in heavy.get('tables', []): 
            context_text += f"[Table]: {t}\n"
            
        # Collect images
        for img in heavy.get('images', []):
            if img not in seen_imgs:
                seen_imgs.add(img)
                retrieved_images.append(img)
                
    # Final Generation
    llm = get_llm(api_key=openai_api_key)
    
    prompt = f"""
    Answer the user question based on the provided context.
    CONTEXT: {context_text[:5000]} 
    QUESTION: {query}
    """
    
    msg_content = [{"type": "text", "text": prompt}]
    
    # Limit Images to top 2 to avoid huge context sizes
    for b64 in retrieved_images[:2]:
        if "," in b64: b64 = b64.split(",")[1]
        msg_content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
        
    response = llm.invoke([HumanMessage(content=msg_content)])
    
    return response.content
