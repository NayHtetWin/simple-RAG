import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil
import time
import logging
import json
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_community.llms import Ollama
# from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_community.document_compressors import FlashrankRerank
from langchain.retrievers import ContextualCompressionRetriever
from starlette.requests import Request

from typing import List 

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("rag_api")


PERSIST_DIRECTORY = "./db_storage"

# --- CONFIGURATION ---
# If running in Docker, we might need to change this to "http://host.docker.internal:11434"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

print(f"--- Connecting to Ollama at: {OLLAMA_BASE_URL} ---", flush=True)

# llm = Ollama(model="mistral")
llm = Ollama(base_url=OLLAMA_BASE_URL, model="mistral")

# Use Ollama for embeddings - no threading issues!
# embedding_function = OllamaEmbeddings(model="nomic-embed-text")
embedding_function = OllamaEmbeddings(base_url=OLLAMA_BASE_URL, model="nomic-embed-text")

vector_db = None
compressor = None

app = FastAPI()

@app.middleware("http")
async def log_performance_metrics(request: Request, call_next):
    # 1. Start the timer
    start_time = time.time()
    
    # 2. Process the request (call the actual endpoint)
    response = await call_next(request)
    
    # 3. Stop the timer
    process_time = time.time() - start_time
    
    # 4. Log the metrics
    logger.info(f"Endpoint: {request.url.path} | Method: {request.method} | Latency: {process_time:.4f}s")
    
    return response

@app.on_event("startup")
async def startup_event():
    """Initialize vector database and reranker on startup"""
    global vector_db, compressor
    print("--- Initializing vector database with Ollama embeddings ---", flush=True)
    vector_db = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embedding_function
    )
    print("--- Vector DB ready ---", flush=True)
    
    print("--- Loading FlashRank model ---", flush=True)
    compressor = FlashrankRerank(top_n=5)
    print("--- FlashRank ready ---", flush=True)



# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    prompt: str

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    try:
        if vector_db is None:
            raise HTTPException(status_code=500, detail="Models not loaded yet")
        
        processed_files = []
        total_chunks = 0

        # LOOP through every uploaded file
        for file in files:
            # 1. Save temp file
            temp_filename = f"temp_{file.filename}"
            with open(temp_filename, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            try:
                print(f"Processing {file.filename}...", flush=True)
                loader = PDFPlumberLoader(temp_filename)
                data = loader.load()
                
                # 2. Clean & Tag Metadata
                # Critical: Ensure every chunk knows its source filename
                for doc in data:
                    doc.metadata["source"] = file.filename
                    content = doc.page_content.replace('\x00', '')
                    doc.page_content = content

                # 3. Split
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents(data)
                
                # 4. Add to DB (Incremental addition)
                vector_db.add_documents(chunks)
                
                processed_files.append(file.filename)
                total_chunks += len(chunks)
                
            finally:
                # Always clean up the temp file, even if errors occur
                if os.path.exists(temp_filename):
                    os.remove(temp_filename)
        
        return {
            "status": "success", 
            "files_processed": processed_files, 
            "total_chunks_added": total_chunks
        }

    except Exception as e:
        print(f"Error in upload: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat")
async def chat(request: QueryRequest):
    try:
        if vector_db is None or compressor is None:
            raise HTTPException(status_code=500, detail="Database or reranker not initialized")

        logger.info(f"Processing query: {request.prompt}")
        start_time = time.time()

        # STEP 1 & 2: RETRIEVAL + RERANKING with LangChain FlashRank
        # Create compression retriever that combines retrieval + reranking
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vector_db.as_retriever(search_kwargs={"k": 20})
        )
        
        # reranked_docs = compression_retriever.invoke(request.prompt)
        # 2. THE FIX: Use 'await' and 'ainvoke'
        # This releases the event loop so other users can hit the API 
        # while this specific request is calculating the rerank.
        reranked_docs = await compression_retriever.ainvoke(request.prompt)
        
        if not reranked_docs:
            return {"response": "No info found."}
        
        # Take top 5
        # top_docs = reranked_docs[:5]
        top_docs = reranked_docs

        retrieval_time = time.time() - start_time
        
        # Log the results
        logger.info(f"Reranked to {len(top_docs)} documents")

        # STEP 3: GENERATION
        context_text = "\n\n".join([doc.page_content for doc in top_docs])
        
        rag_prompt = f"""
        You are an expert analyst. Answer the question based ONLY on the following context.
        If the context doesn't contain the answer, say "I do not have enough information in the documents."
        
        Context:
        {context_text}
        
        Question: 
        {request.prompt}
        """

        async def generate_stream():
            metadata = {
                "sources": [doc.metadata.get("source", "unknown") for doc in top_docs],
                "retrieval_latency": round(retrieval_time, 4)
            }
            yield json.dumps(metadata) + "\n"

            for chunk in llm.stream(rag_prompt):
                yield chunk 

        return StreamingResponse(generate_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))