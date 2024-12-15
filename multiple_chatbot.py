import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_google_genai import ChatGoogleGenerativeAI
from transformers import pipeline
from huggingface_hub import login
from langchain_ollama import OllamaLLM

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
# google_api_key = os.getenv("GOOGLE_API_KEY")
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure Hugging Face authentication is in place
if huggingface_api_token:
    login(token=huggingface_api_token)
else:
    st.error("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your .env file.")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200, length_function=len)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain(model_name):
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context". 
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    
    # model_mapping = {
    #     "Gemma 2": pipeline("text-generation", model="google/gemma-2-9b"),
    #     "Llama 3.1": pipeline("text-generation", model="meta-llama/Llama-3.1-8B"),
    #     "Mistral": pipeline("text-generation", model="mistralai/Mistral-7B-v0.1"),
    #     "Qwen 2": pipeline("text-generation", model="Qwen/Qwen2-7B")
    # }
    model_mapping = {
        "Gemma 2": OllamaLLM(model="phi3"),
        "Others" : None
        # "Llama 3.1": OllamaLLM(model="llama3.1:8b"),
        # "Mistral": OllamaLLM(model="mistral:7b"),
        # "Qwen 2": OllamaLLM(model="qwen2:7b")
    }
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model_mapping[model_name], chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, model_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain(model_name)
    start_time = time.time()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    elapsed_time = time.time() - start_time
    return response["output_text"], elapsed_time

def main():
    st.set_page_config("Model Comparison Chat with PDF")
    st.header("Compare AI Models Chat with PDF 💬")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    model_name = st.selectbox("Select Model", ["Gemma 2", "Llama 3.1", "Mistral", "Qwen 2"])
    
    if user_question and model_name:
        with st.spinner(f"Getting response from {model_name}..."):
            response, elapsed_time = user_input(user_question, model_name)
            st.write(f"### {model_name} Reply:")
            st.write(response)
            st.write(f"Response Time: {elapsed_time:.2f} seconds")
    
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs Processed Successfully!")

if __name__ == "__main__":
    main()