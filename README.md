# Multi-Modal RAG for financial document processing

Welcome to the **Financial Intelligence Platform**, a multi-modal Retrieval-Augmented Generation (RAG) application for analyzing financial documents and generating insights. This platform enables users to upload documents, perform advanced searches, and interact with an AI-powered chatbot that uses financial data to answer questions.

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [API Endpoints](#api-endpoints)
- [Contributing](#contributing)
- [License](#license)

---

## Overview
The Financial Intelligence Platform integrates document processing, embeddings-based search, and large language models (LLMs) to provide:

- **Automated document indexing** for PDFs.
- **AI-driven search** for financial insights.
- **Interactive chatbot** using advanced language models.

This application uses **streamlit** for the frontend and **FastAPI** for the backend.

---

## Features
- **Document Upload and Indexing**: Upload PDF documents for automatic processing and indexing.
- **Multi-Modal Retrieval-Augmented Generation (RAG)**: Search across text, tables, and images.
- **Chatbot for Query Responses**: AI-powered responses based on document context.
- **Pinecone Integration**: Vector-based search using embeddings from HuggingFace.
- **Agent Report Generation (coming soon)**.

---

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: FastAPI
- **AI Models**: OpenAI GPT-4o, Llama 3.2-11B Vision-Instruct
- **Embeddings**: HuggingFace `sentence-transformers/all-mpnet-base-v2`
- **Vector Store**: Pinecone
- **Document Partitioning**: unstructured partition library

---

## Installation
### Prerequisites
Ensure you have the following installed:
- Python 3.8+
- Pip

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/financial-intelligence-platform.git
   cd financial-intelligence-platform
   ```
2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows, use venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set environment variables for API keys:
   ```bash
   export OPENAI_API_KEY=your_openai_api_key
   export PINECONE_API_KEY=your_pinecone_api_key
   export LLAMA_API_KEY=your_llama_api_key
   ```

5. Start the backend server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
6. Launch the frontend:
   ```bash
   streamlit run frontend.py
   ```

---

## Usage
### Uploading Documents
1. Navigate to the **RAG** tab in the Streamlit interface.
2. Upload PDF documents using the file uploader.
3. View and interact with the indexed content.

### Querying Documents
1. Ask questions related to uploaded documents in the chatbot input.
2. Stream real-time AI-generated responses based on your query.

---

## API Endpoints
### `/rag/upload`
**Method**: POST
- **Description**: Upload and process documents.
- **Parameters**:
  - `files`: List of files (PDFs).
  - `document_type`: Type of document (e.g., `pdf`).

### `/rag/search`
**Method**: POST
- **Description**: Query indexed documents.
- **Parameters**:
  - `query`: Search query string.

---

## Contributing
Contributions are welcome! Please fork the repository and create a pull request. Ensure code quality and tests accompany significant changes.

### Development Setup
To contribute:
1. Clone the repository.
2. Follow the installation steps.
3. Use feature branches for new development.

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgements
- [OpenAI](https://openai.com) for GPT models.
- [HuggingFace](https://huggingface.co) for transformer-based embeddings.
- [Streamlit](https://streamlit.io) for UI development.
- [Pinecone](https://pinecone.io) for vector storage.

