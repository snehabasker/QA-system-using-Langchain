import streamlit as st
import os
from qa_langchain import download_squad_data, build_vectorstore, answer_question

st.set_page_config(
    page_title="QA System using Langchain",
    page_icon="🤖",
    layout="wide"
)

st.title("Question Answering System")
st.markdown("Powered by Langchain + FLAN-T5 + SQuAD Dataset")

st.sidebar.header("About")
st.sidebar.markdown("""
This QA system uses:
- Langchain for orchestration
- FLAN-T5 for text generation
- FAISS for vector search
- SQuAD dataset as knowledge base
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Built by: Sneha Basker")
st.sidebar.markdown("[GitHub](https://github.com/snehabasker)")

if 'vectorstore' not in st.session_state:
    with st.spinner("Downloading SQuAD dataset (first time only)..."):
        data_path = download_squad_data()
    
    with st.spinner("Building vector store (takes 1-2 minutes)..."):
        st.session_state.vectorstore = build_vectorstore(data_path)
    
    st.success("System ready!")

st.header("Ask a Question")

st.markdown("Example questions:")
examples = [
    "What is machine learning?",
    "Who invented the telephone?",
    "What is the capital of France?",
    "When did World War 2 end?"
]

cols = st.columns(2)
for i, example in enumerate(examples):
    with cols[i % 2]:
        if st.button(example, key=f"ex_{i}"):
            st.session_state.query = example

query = st.text_input(
    "Enter your question:",
    value=st.session_state.get('query', ''),
    placeholder="e.g., What is artificial intelligence?"
)

if st.button("Get Answer", type="primary", use_container_width=True):
    if query:
        with st.spinner("Thinking..."):
            answer, docs = answer_question(query, st.session_state.vectorstore)
        
        st.markdown("---")
        st.markdown("Answer:")
