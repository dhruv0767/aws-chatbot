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
from langchain_aws import BedrockLLM

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

def create_vector_store():
    loader = PyPDFDirectoryLoader("Data")
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=256,
        chunk_overlap=200
    )
    docs = text_splitter.split_documents(documents)
    vectorstore_faiss = FAISS.from_documents(
        docs,
        bedrock_embeddings
    )
    vectorstore_faiss.save_local("faiss_index")
    return vectorstore_faiss

def load_vector_store():
    if not os.path.exists("faiss_index"):
        return create_vector_store()
    return FAISS.load_local(
        "faiss_index", 
        bedrock_embeddings, 
        allow_dangerous_deserialization=True
    )

def get_llm():
    return BedrockLLM(
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
    answer = qa.invoke({"query": query})
    return answer['result']

def main():
    st.set_page_config("Chat PDF")
    st.header("Welcome to Dhruv's Personal Chatbot!")
    
    # Add key parameter to track the input state
    user_question = st.text_input("Ask a Question which you would like to know about Dhruv!", key="question_input", on_change=None)

    # Check for either button click or Enter key press (when input changes)
    if st.button("Output") or user_question:
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
