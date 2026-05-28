import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# 1. API KEYS & CONFIG
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "pdf-mcp-db")  # Default index name

# OpenAI Configuration
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.4-mini")

PDF_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Prior Authorization Documentation Guide.pdf")
LOCAL_STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "local_store.pkl")

# 2. MODEL INITIALIZATION
def get_llm(api_key: str, temperature=0):
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be provided dynamically.")
        
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        api_key=api_key,
        model=OPENAI_MODEL,
        temperature=temperature,
        streaming=False
    )

def get_embeddings():
    # IntFloat E5-Base v2 (768 dim) is used to match Pinecone
    return HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
