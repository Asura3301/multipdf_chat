import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from html_templates import css, bot_template, user_template
from langchain import hub
import os
import getpass


#------------------------------------ 
# setup environment variables

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
    
#------------------------------------

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
# helper function for conversation chain

def get_conversation_chain(vectorstore):
    
    # retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", 
        search_kwargs={"k":3},
    )
    
    llm = ChatGroq(
        temperature=0.4, 
        model_name="llama-3.3-70b-versatile", 
        max_tokens=1024
    )
    # Contextualize question
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, just "
        "reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    # Create a history-aware retriever
    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever,
        contextualize_q_prompt,
    )
    
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(
        llm,
        retrieval_qa_chat_prompt,
    )
    
    rag_chain = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=combine_docs_chain,
    )

    return rag_chain

#------------------------------------
# Answer user question

def handle_user_input(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.write(response)
    
    
    
#------------------------------------
def main():
    # init environment variables
    load_dotenv()
    
    st.set_page_config(
        page_title="MultiPDF Chatbot", 
        page_icon="🤖", 
        layout="centered", 
        initial_sidebar_state="expanded"
        )
    
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    st.header("MultiPDF Chatbot 🤖")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)
    
    st.write(user_template.replace("{{MSG}}", ""), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", "How can I help you?"), unsafe_allow_html=True)
    
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
            vectorstore = get_vector_store(text_chunks)

            # conversation chain
            st.session_state.conversation = get_conversation_chain(vectorstore)

    st.session_state.conversation

if __name__=="__main__":
    main()