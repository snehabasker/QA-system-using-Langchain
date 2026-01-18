import os
from transformers import pipeline
import requests
import json

generator = pipeline("text2text-generation", model="google/flan-t5-base")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def download_squad_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    contexts_path = os.path.join(data_dir, "squad_contexts.txt")
    
    if os.path.exists(contexts_path):
        print("SQuAD data already exists.")
        return contexts_path
    
    print("Downloading SQuAD dataset...")
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json"
    response = requests.get(url)
    squad_data = response.json()
    
    print("Extracting contexts...")
    contexts = []
    for article in squad_data['data']:
        for paragraph in article['paragraphs']:
            context = paragraph['context'].replace('\n', ' ').strip()
            contexts.append(context)
    
    print(f"Writing {len(contexts)} contexts to file...")
    with open(contexts_path, 'w', encoding='utf-8') as out_file:
        for context in contexts:
            out_file.write(context + '\n\n')
    
    print(f"Saved to {contexts_path}")
    return contexts_path


def build_vectorstore(data_path):
    print("Loading and chunking documents...")
    
    with open(data_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(content)
    
    chunks = chunks[:100]
    
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    print(f"Chunked into {len(documents)} segments.")
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    return vectorstore


def answer_question(query, vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n".join(doc.page_content for doc in docs)
    
    prompt = (
        "Use the CONTEXT below to answer the QUESTION.\n"
        "If answer not found, say 'Not found in source.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
    )
    
    result = generator(prompt, max_length=256)
    answer = result[0]['generated_text']
    
    return answer.strip(), docs


def main():
    data_path = download_squad_data()
    
    print("Building vector store (this may take a minute)...")
    vectorstore = build_vectorstore(data_path)
    
    print("\nREADY: Retrieval-Augmented Q&A System")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter question: ")
        if query.lower() in {"exit", "quit", ""}:
            print("Session ended.")
            break
        
        answer, docs = answer_question(query, vectorstore)
        print(f"\nAnswer:\n{answer}\n")
        print(f"Sources: {len(docs)} relevant passages found\n")
        print("-" * 50)


if __name__ == "__main__":
    main()
```

And requirements.txt:
```
streamlit
langchain
langchain-community
faiss-cpu
sentence-transformers
transformers
torch
requests
