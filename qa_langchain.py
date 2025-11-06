import os
from transformers import pipeline

# Initialize text generation pipeline globally
generator = pipeline("text2text-generation", model="google/flan-t5-base")

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter




def build_vectorstore(data_path):
    print("Loading and chunking documents...")
    loader = TextLoader(data_path, encoding="utf-8")
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = splitter.split_documents(documents[:20])  # Limit for faster processing
    print(f"Chunked into {len(texts)} segments.")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore
generator = pipeline("text2text-generation", model="google/flan-t5-base")


def answer_question(query, vectorstore):
    retriever = vectorstore.as_retriever()
    docs = retriever.invoke(query)
    context = "\n".join(doc.page_content for doc in docs)
    prompt = (
        "Use the CONTEXT below to answer the QUESTION.\n"
        "If answer not found, say 'Not found in source.'\n\n"
        f"CONTEXT:\n{context}\n\nQUESTION: {query}\nANSWER:"
    )
    result = generator(prompt, max_length=256)
    answer = result[0]['generated_text']
    return answer.strip()


def main():
    data_path = "data/squad_contexts.txt"
    vectorstore = build_vectorstore(data_path)
    print("READY: Retrieval-Augmented Q&A Fully Local (Type 'exit' to quit)")
    while True:
        query = input("\nEnter question: ")
        if query.lower() in {"exit", "quit"}:
            print("Session ended.")
            break
        answer = answer_question(query, vectorstore)
        print(f"\nAnswer:\n{answer}")

if __name__ == "__main__":
    main()
