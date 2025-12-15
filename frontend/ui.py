import streamlit as st
import requests
import json
import os

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="RAG Portfolio Architect", layout="wide")
st.title("📄 Simple RAG System")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Document Ingestion")
    uploaded_files = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            with st.spinner("Ingesting documents..."):
                try:
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

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your document..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            with requests.post(
                f"{BACKEND_URL}/chat", 
                json={"prompt": prompt}, 
                stream=True
            ) as r:
                r.raise_for_status()
                
                for line in r.iter_lines():
                    if line:
                        decoded_line = line.decode('utf-8')
                        
                        if decoded_line.startswith('{') and "sources" in decoded_line:
                            meta = json.loads(decoded_line)
                            with st.expander("Debugging Metrics"):
                                st.write(f"Retrieval Latency: {meta.get('retrieval_latency')}s")
                                st.write(f"Sources: {meta.get('sources')}")
                        else:
                            full_response += decoded_line
                            message_placeholder.markdown(full_response + "▌")
                            
                message_placeholder.markdown(full_response)
                
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")

        st.session_state.messages.append({"role": "assistant", "content": full_response})