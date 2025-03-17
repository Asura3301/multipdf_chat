import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS




# helper functions for extracting text from pdf
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#------------------------------------

# helper function for splitting text into chunks
def get_text_chunks(text, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks

#------------------------------------

# helper function for downloading embeddings
def download_embeddings():
    embed_model = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=embed_model)
    return embeddings

# get vector store

def get_vector_store(text_chunks):
    embeddings = download_embeddings()
    vector_store = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings,
    )
    return vector_store


#------------------------------------

def main():
    # init environment variables
    load_dotenv()
    
    st.set_page_config(page_title="MultiPDF Chatbot", page_icon="🤖", layout="centered", initial_sidebar_state="expanded")
    
    st.header("MultiPDF Chatbot 🤖")
    st.text_input("Ask a question about your documents:")
    
    with st.sidebar:
        st.subheader("Document List")
        pdf_docs = st.file_uploader("Upload Document", type=["pdf"], accept_multiple_files=True)
        if st.button("Upload"):
            st.spinner("Processing...")
            # get pdf text
            raw_text = get_pdf_text(pdf_docs) 
            
            # get text chunks
            text_chunks = get_text_chunks(raw_text, chunk_size=512, chunk_overlap=64)
            st.write(text_chunks)
            
            # create vectore store with embeddings
            vectorestore = get_vector_store(text_chunks)
        
        
        
if __name__=="__main__":
    main()