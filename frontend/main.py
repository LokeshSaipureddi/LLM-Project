import streamlit as st
import requests
from pathlib import Path
import base64

# Backend endpoints
UPLOAD_ENDPOINT = "http://localhost:8000/rag/upload"
SEARCH_ENDPOINT = "http://localhost:8000/rag/search"

# Page config
st.set_page_config(page_title="Financial Chatbot", layout="wide")

# Initialize session state for chat history, uploaded files, and temporary messages
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "uploaded_files" not in st.session_state:
    st.session_state["uploaded_files"] = []
if "page" not in st.session_state:
    st.session_state["page"] = "RAG"
if "upload_message" not in st.session_state:
    st.session_state["upload_message"] = None  # For showing the upload success message

# Sidebar for navigation and file uploads
with st.sidebar:
    st.title("🧠 Financial Intelligence Platform")
    st.session_state["page"] = st.selectbox("Select Mode:", ["RAG", "Agent"], key="mode_selector")

    # Sidebar for uploading/viewing PDFs (RAG only)
    if st.session_state["page"] == "RAG":
        st.title("📁 Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDFs:", type=["pdf"], accept_multiple_files=True, key="file_uploader"
        )

        document_type = st.selectbox("Document Type:", ["pdf"], index=0)

        if st.button("Upload Documents"):
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    # Save uploaded files temporarily to display in the sidebar
                    temp_path = Path("/tmp") / uploaded_file.name
                    temp_path.write_bytes(uploaded_file.getvalue())

                    # Store file for display and processing
                    st.session_state["uploaded_files"].append(temp_path)

                    # Send file to backend
                    files = {"files": (uploaded_file.name, uploaded_file.getvalue())}
                    data = {"document_type": document_type}
                    response = requests.post(UPLOAD_ENDPOINT, files=files, data=data)

                    if response.status_code == 200:
                        st.success(f"Uploaded: {uploaded_file.name}")

        st.write("### Uploaded PDFs")
        if st.session_state["uploaded_files"]:
            selected_pdf = st.selectbox(
                "Select a PDF to view:", [file.name for file in st.session_state["uploaded_files"]]
            )

            if selected_pdf:
                for file in st.session_state["uploaded_files"]:
                    if file.name == selected_pdf:
                        with open(file, "rb") as f:
                            pdf_display = f"""<iframe src="data:application/pdf;base64,{base64.b64encode(f.read()).decode()}" width="100%" height="400px"></iframe>"""
                            st.markdown(pdf_display, unsafe_allow_html=True)

# Main Content Area
if st.session_state["page"] == "RAG":
    st.title("💬 Financial RAG Chatbot")

    # Display chat history
    for chat in st.session_state["chat_history"]:
        st.chat_message("user").write(chat['user'])
        with st.chat_message("assistant"):
            st.write(chat['bot'])

    # Input for user query
    query = st.chat_input("Ask a question about your documents")

    if query:
        # Append the query to the chat history immediately
        st.session_state["chat_history"].append({"user": query, "bot": ""})

        # Display the user's message
        st.chat_message("user").write(query)

        # Create a placeholder for assistant response
        assistant_message = st.chat_message("assistant")
        response_container = assistant_message.empty()

        try:
            response = requests.post(SEARCH_ENDPOINT, data={"query": query}, stream=True)

            # Collect and stream response incrementally
            full_response = ""
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    decoded_chunk = chunk.decode('utf-8')
                    full_response += decoded_chunk
                    response_container.write(full_response)

            # Update the last chat entry with the full response
            st.session_state["chat_history"][-1]["bot"] = full_response

        except Exception as e:
            assistant_message.write(f"An error occurred: {e}")

elif st.session_state["page"] == "Agent":
    st.title("🤖 Financial Agent")
    st.text("This page is under construction. Add agent-specific UI elements here.")
