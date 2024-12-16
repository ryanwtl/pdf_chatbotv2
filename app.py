import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os
import time
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from huggingface_hub import login, InferenceClient
from langchain_ollama import OllamaLLM

# Load environment variables from .env file
load_dotenv()

# Retrieve API keys from environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
huggingface_api_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Ensure Hugging Face authentication is in place
if huggingface_api_token:
    login(token=huggingface_api_token)
else:
    st.error("Hugging Face API token not found. Please set HUGGINGFACEHUB_API_TOKEN in your .env file.")

# Initialize Hugging Face InferenceClient
hf_client = InferenceClient(api_key=huggingface_api_token)

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
    # embeddings = hf_client.embeddings(model="sentence-transformers/all-MiniLM-L6-v2") # Use embeddings method
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
    
    model_mapping = {
        "Llama 3.3-70B-Instruct": "llama-3.3-70B-Instruct",  # Placeholder for special handling
        "Others": None
    }
    
    if model_name == "Llama 3.3-70B-Instruct":
        return model_mapping[model_name]
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model_mapping[model_name], chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, model_name):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # embeddings = hf_client.embeddings(model="sentence-transformers/all-MiniLM-L6-v2") # Use embeddings method
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    
    if model_name == "Llama 3.3-70B-Instruct":
        messages = [{"role": "user", "content": user_question}]
        stream = hf_client.chat.completions.create(
            model="meta-llama/Llama-3.3-70B-Instruct",
            messages=messages,
            max_tokens=500,
            stream=True
        )
        start_time = time.time()
        response_text = "".join(chunk.choices[0].delta.content for chunk in stream)
        elapsed_time = time.time() - start_time
        return response_text, elapsed_time
    else:
        chain = get_conversational_chain(model_name)
        start_time = time.time()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        elapsed_time = time.time() - start_time
        return response["output_text"], elapsed_time

def main():
    st.set_page_config("Model Comparison Chat with PDF")
    st.header("Compare AI Models Chat with PDF 💬")
    
    user_question = st.text_input("Ask a Question from the PDF Files")
    model_name = st.selectbox("Select Model", ["Llama 3.3-70B-Instruct", "Others"])
    
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
