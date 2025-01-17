import json
import os
import sys
import boto3
import streamlit as st
import numpy as np

# LangChain Community Imports
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.llms.bedrock import Bedrock

# LangChain Core Imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Initialize Bedrock Client
bedrock = boto3.client(service_name="bedrock-runtime")
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock
)

def data_ingestion():
    loader = PyPDFDirectoryLoader("Data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=1000
    )
    docs = text_splitter.split_documents(documents)
    return docs

def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def load_vector_store():
    return FAISS.load_local(
        "faiss_index", 
        bedrock_embeddings, 
        allow_dangerous_deserialization=True
    )

def get_llm():
    return Bedrock(
        model_id="anthropic.claude-instant-v1",
        client=bedrock,
        model_kwargs={'max_tokens_to_sample': 512}
    )

prompt_template = """
H:You are a highly intelligent and professional chatbot designed to answer questions about Dhruv Shah, his projects, career, and expertise. 
Use the given context to deliver detailed, concise, and professional answers. Always conclude by suggesting three additional related questions 
to engage the user further. The questions should be in bulleted points. If the information is unavailable, say "I don't know" without speculating.

<context>
{context}
</context>

Question: {question}

A:"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity", 
            search_kwargs={"k": 3}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using AWS Bedrock💁")
    
    user_question = st.text_input("Ask a Question from the PDF Files")

    with st.sidebar:
        st.title("Update Or Create Vector Store:")
        if st.button("Vectors Update"):
            with st.spinner("Processing..."):
                try:
                    docs = data_ingestion()
                    get_vector_store(docs)
                    st.success("Vector store updated successfully!")
                except Exception as e:
                    st.error(f"Error updating vector store: {str(e)}")

    if st.button("Output"):
        if not user_question:
            st.warning("Please enter a question first.")
            return
            
        with st.spinner("Processing..."):
            try:
                faiss_index = load_vector_store()
                llm = get_llm()
                response = get_response_llm(llm, faiss_index, user_question)
                st.write(response)
                st.success("Response generated successfully!")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()
