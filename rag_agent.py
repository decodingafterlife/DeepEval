# rag_agent.py

import json
import time
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import config

def create_rag_chain():
    """Builds and returns a LangChain RAG chain."""
    print("--- Building RAG Agent ---")
    loader = PyPDFLoader(config.SOURCE_DOCUMENT)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    embeddings = OllamaEmbeddings(model=config.LC_EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(splits, embeddings)
    retriever = vector_store.as_retriever()
    llm = OllamaLLM(model=config.LC_LLM_MODEL)
    
    prompt = ChatPromptTemplate.from_template(
        "Please answer the following question based only on the provided context.\n\n"
        "Context:\n{context}\n\n"
        "Question: {input}"
    )
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("✅ RAG Agent built successfully.")
    return retrieval_chain

def generate_rag_responses(input_goldens_path, output_goldens_path):
    """
    Loads goldens, runs them through the RAG agent, and saves the results.
    """
    rag_chain = create_rag_chain()
    if rag_chain is None:
        print("Exiting due to RAG chain setup failure.")
        return

    with open(input_goldens_path, 'r', encoding='utf-8') as f:
        goldens_data = json.load(f)

    print(f"\n--- Running RAG agent on {len(goldens_data)} test cases ---")
    for i, item in enumerate(goldens_data):
        question = item.get("input")
        print(f"  - Processing question {i+1}/{len(goldens_data)}...")
        
        response = rag_chain.invoke({"input": question})
        
        answer = response.get("answer", "No answer found.")
        context_docs = response.get("context", [])
        retrieved_context = [doc.page_content for doc in context_docs]

        item["actual_output"] = answer
        item["retrieval_context"] = retrieved_context

    with open(output_goldens_path, 'w') as f:
        json.dump(goldens_data, f, indent=4)
    
    print(f"✅ RAG agent responses saved to '{output_goldens_path}'.")