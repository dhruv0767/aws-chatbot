import json
import os
import sys
import boto3
import streamlit as st
import numpy as np

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
