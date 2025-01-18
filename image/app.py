from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import boto3
from langchain_aws import BedrockEmbeddings, BedrockLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Initialize AWS Bedrock Client
bedrock = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1",  
    aws_access_key_id="AKIA4ZPZU2D4VOMTJP7C",
    aws_secret_access_key="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"
)

# Initialize Bedrock embeddings
try:
    bedrock_embeddings = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        client=bedrock
    )
    logger.info("Bedrock embeddings initialized successfully.")
except Exception as e:
    logger.error(f"Error initializing Bedrock embeddings: {e}")
    raise RuntimeError("Error initializing Bedrock embeddings.")

# Prompt template for the chatbot
PROMPT = PromptTemplate(
    template="""
    H:You are a highly intelligent and professional chatbot designed to answer questions about Dhruv Shah, his projects, career, and expertise. 
    Use the given context to deliver detailed, concise, and professional answers. Always conclude by suggesting three additional related questions 
    to engage the user further. The questions should be in bulleted points. If the information is unavailable, say "I don't know" without speculating.

    <context>
    {context}
    </context>

    Question: {question}

    A:""",
    input_variables=["context", "question"]
)

# Pydantic model for request validation
class QueryRequest(BaseModel):
    question: str

# Function to load FAISS index
def load_vector_store():
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError("FAISS index not found. Ensure 'faiss_index' directory exists and contains valid files.")
    return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

@app.post("/chat/")
def chat_with_bot(request: QueryRequest):
    logger.info(f"Received question: {request.question}")
    try:
        # Load FAISS index
        vectorstore = load_vector_store()
        logger.info("FAISS index loaded successfully.")

        # Initialize Bedrock LLM
        llm = BedrockLLM(
            model_id="anthropic.claude-instant-v1",
            client=bedrock,
            model_kwargs={'max_tokens_to_sample': 512}
        )
        logger.info("LLM initialized successfully.")

        # Retrieve documents
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(request.question)
        logger.info(f"Retrieved documents: {docs}")

        # Setup RetrievalQA
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        # Generate response
        answer = qa.invoke({"query": request.question})
        logger.info(f"Generated response: {answer['result']}")
        return {"response": answer['result']}
    except Exception as e:
        logger.error(f"Error during chatbot processing: {e}")
        raise HTTPException(status_code=500, detail="Error during chatbot processing")

@app.get("/")
def read_root():
    return {"message": "Welcome to Dhruv's Chatbot API!"}
