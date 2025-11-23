import streamlit as st
import requests
import json
import os

# --- CONFIGURATION ---
# Use the Docker service name "rag-api" to talk to the backend
# When running locally outside docker, use localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Portfolio Architect", layout="wide")
st.title("📄 Simple RAG System")

# Session State to hold chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for File Upload
with st.sidebar:
    st.header("Document Ingestion")
    # CHANGE: accept_multiple_files=True
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"): # Add a button so it doesn't trigger on every file select
            with st.spinner("Ingesting documents..."):
                try:
                    # Prepare list of files for the API
                    # format: [('files', (filename, file_obj, type)), ...]
                    files_payload = [
                        ("files", (file.name, file, "application/pdf")) 
                        for file in uploaded_files
                    ]
                    
                    response = requests.post(f"{BACKEND_URL}/upload", files=files_payload)
                    
                    if response.status_code == 200:
                        data = response.json()
                        st.success(f"Success! Processed {len(data['files_processed'])} files.")
                    else:
                        st.error(f"Upload failed: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Logic
if prompt := st.chat_input("Ask about your document..."):
    # 1. Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Display assistant response (Streaming)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Connect to the backend with streaming enabled
            with requests.post(
                f"{BACKEND_URL}/chat", 
                json={"prompt": prompt}, 
                stream=True
            ) as r:
                r.raise_for_status()
                
                # Process the stream line by line
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        # Check if it's the Metadata JSON (first line)
                        if decoded_line.startswith('{') and "sources" in decoded_line:
                            meta = json.loads(decoded_line)
                            # Optional: Display retrieval time or sources in an expander
                            with st.expander("Debugging Metrics"):
                                st.write(f"Retrieval Latency: {meta.get('retrieval_latency')}s")
                                st.write(f"Sources: {meta.get('sources')}")
                        else:
                            # It's a text token
                            full_response += decoded_line
                            message_placeholder.markdown(full_response + "▌")
                            
                message_placeholder.markdown(full_response)
                
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")

        # Save response to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})