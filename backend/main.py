import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import List, Optional
from rag_module import FinancialRAGSystem
from agent_module import FinancialReportAgent
import asyncio
import json

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Financial Intelligence Platform")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize systems
rag_system = FinancialRAGSystem()

@app.post("/rag/upload")
async def upload_documents(
    files: List[UploadFile] = File(...),
    document_type: str = Form(...)
):
    """
    Upload documents for RAG system

    :param files: List of uploaded files
    :param document_type: Type of documents (pdf/image)
    :return: Upload status
    """
    try:
        # Save files temporarily
        file_paths = []
        for file in files:
            temp_path = f"/tmp/{file.filename}"
            with open(temp_path, "wb") as buffer:
                buffer.write(await file.read())
            file_paths.append(temp_path)

        # Process documents based on type
        if document_type == "pdf":
            for file_path in file_paths:
                rag_system.process_and_index(file_path)

        return {"status": "success", "message": "Documents uploaded and processed"}

    except Exception as e:
        return {"status": "error", "message": str(e)}

async def generate_streaming_response(query):
    """
    Generate a streaming response for the RAG query
    """
    # Perform multi-modal search
    full_response = rag_system.search(query)
    
    # Stream the response word by word
    for word in full_response.split():
        yield f"{word} "
        await asyncio.sleep(0.1)  # Add a small delay between words

@app.post("/rag/search")
async def rag_index(query: str = Form(...)):
    try:
        response = rag_system.search(query)
        return StreamingResponse(generate_streaming_response(response), media_type="text/plain")
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
