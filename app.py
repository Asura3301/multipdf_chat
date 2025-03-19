import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain import hub
from html_templates import css, bot_template, user_template
import os
import getpass
from config import *  # Import all configuration settings


#------------------------------------ 
# setup environment variables

if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")
HUGGINGFACEHUB_API_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
#------------------------------------

def get_pdf_text(pdf_docs):
    """Extract text from multiple PDF documents."""
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Document: {pdf.name}, Page {page_num+1} ---\n"
                    text += page_text
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
    return text

#------------------------------------

def get_text_chunks(text, chunk_size, chunk_overlap):
    """Split text into chunks of a given size and overlap."""
    
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap,
        length_function=len
        )
    chunks = text_splitter.split_text(text)
    return chunks

#------------------------------------

@st.cache_resource
def download_embeddings():
    """Download and cache embeddings from HuggingFace."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        return embeddings
    except Exception as e:
        st.error(f"Error downloading embeddings: {str(e)}")
        return None


def get_vector_store(text_chunks):
    """Create a vector store from text chunks."""
    if not text_chunks:
        st.error("No text chunks provided!")
        return None
    
    embeddings = download_embeddings()
    if embeddings is None:
        st.error("Failed to download embeddings!")
        return None
    
    try:
        vector_store = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings,
        )
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {str(e)}")
        return None


#------------------------------------


def get_conversation_chain(vectorstore):
    """Create a conversation chain."""
    if not vectorstore:
        st.error("No vector store provided!")
        return None
    
    try:
        # retriever
        retriever = vectorstore.as_retriever(
            search_type=RETRIEVER_SEARCH_TYPE, 
            search_kwargs={"k": RETRIEVER_K},
        )
        
        # init LLM
        llm = ChatGroq(
            temperature=DEFAULT_TEMPERATURE, 
            model_name=DEFAULT_MODEL, 
            max_tokens=DEFAULT_MAX_TOKENS
        )
        
        # Another way to use LLMs via HuggingFace Hub
        # llm = HuggingFaceEndpoint(
        #     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        #     task="text-generation",
        #     max_new_tokens=100,
        #     do_sample=False,
        # )
        
        # Contextualize question prompt
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", "Given a chat history and the latest user question "
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, just "
                "reformulate it if needed and otherwise return it as is."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        # Create a history-aware retriever
        history_aware_retriever = create_history_aware_retriever(
            llm, 
            retriever,
            contextualize_q_prompt,
        )
        
        # get retrieval qa chat prompt from hub
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        
        # create combine docs chain
        combine_docs_chain = create_stuff_documents_chain(
            llm,
            retrieval_qa_chat_prompt,
        )
        
        # create retrieval chain
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=combine_docs_chain,
        )
        
        return rag_chain
    
    except Exception as e:
        st.error(f"Error creating conversation chain: {str(e)}")
        return None

#------------------------------------
# Answer user question

def handle_user_input(user_question):
    """Process user input and generate AI response."""
    if st.session_state.conversation is None:
        st.error("Please upload documents first!")
        return
    
    try:
        # Format chat history for the chain
        formatted_chat_history = st.session_state.chat_history
        
        # Invoke the chain
        with st.spinner(""):
            response = st.session_state.conversation.invoke(
                {
                    "input": user_question,
                    "chat_history": formatted_chat_history
                }
            )
        
        
        # Add user question and response to chat history
        st.session_state.chat_history.append(("human", user_question))
        st.session_state.chat_history.append(("ai", response["answer"]))
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        # Add error message to chat history
        st.session_state.chat_history.append(("human", user_question))
        st.session_state.chat_history.append(("ai", f"Error: {str(e)}"))
     
    
#------------------------------------
def main():
    # init environment variables
    load_dotenv()
    
    st.set_page_config(
        page_title=APP_TITLE, 
        page_icon=APP_ICON, 
        layout="centered", 
        initial_sidebar_state="expanded"
    )
    
    st.write(css, unsafe_allow_html=True)
    
    # Initialize session states
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = ""
    
    # Page header with logo and title
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image(APP_LOGO, width=LOGO_WIDTH)
    with col2:
        st.title(APP_TITLE)
    
    st.markdown("---")
    
    # Check if there's a pending question to process
    if st.session_state.user_question:
        question = st.session_state.user_question
        st.session_state.user_question = ""  # Clear the question
        handle_user_input(question)
    
    # Sidebar for document upload and processing
    with st.sidebar:
        st.header("Document Settings")
        
        # File uploader
        pdf_docs = st.file_uploader(
            "Upload your PDFs",
            type=["pdf"],
            accept_multiple_files=True,
            help="Select one or more PDF files to analyze"
        )
        
        # Processing options
        st.subheader("Processing Options")
        chunk_size = st.slider("Chunk Size", 
                              min_value=MIN_CHUNK_SIZE, 
                              max_value=MAX_CHUNK_SIZE, 
                              value=DEFAULT_CHUNK_SIZE, 
                              step=CHUNK_SIZE_STEP,
                              help="Size of text chunks for processing")
        
        chunk_overlap = st.slider("Chunk Overlap", 
                                 min_value=MIN_CHUNK_OVERLAP, 
                                 max_value=MAX_CHUNK_OVERLAP, 
                                 value=DEFAULT_CHUNK_OVERLAP, 
                                 step=CHUNK_OVERLAP_STEP,
                                 help="Overlap between consecutive chunks")
        
        # Model selection
        model_name = st.selectbox(
            "Select LLM Model",
            [DEFAULT_MODEL] + ALTERNATIVE_MODELS,
            index=0,
            help="Choose the language model to use"
        )
        
        # Process button
        if st.button("Process Documents"):
            if not pdf_docs:
                st.error("Please upload at least one PDF document!")
            else:
                with st.spinner("Processing your documents..."):
                    # Extract text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    
                    # Create vector store
                    vectorstore = get_vector_store(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    
                    # Reset chat history when new documents are processed
                    st.session_state.chat_history = []
                    
                    st.session_state.processing_status = f"✅ Processed {len(pdf_docs)} documents with {len(text_chunks)} chunks"
                    st.success("Documents processed successfully! You can now ask questions.")
        
        if st.session_state.processing_status:
            st.info(st.session_state.processing_status)
            
        # Document list
        if pdf_docs:
            st.subheader("Uploaded Documents")
            for i, doc in enumerate(pdf_docs):
                st.text(f"{i+1}. {doc.name}")
        
        st.markdown("---")
        st.markdown("### About")
        st.markdown(ABOUT_TEXT)
    
    # Main chat interface
    if st.session_state.conversation is None:
        st.info("👈 Please upload and process documents using the sidebar to start chatting!")
        
        # Sample questions placeholder
        st.markdown("### Sample Questions You Can Ask:")
        for q in SAMPLE_QUESTIONS:
            st.markdown(f"- *{q}*")
    else:
        # Chat history container with dynamic height
        chat_history_length = len(st.session_state.chat_history)
        
        # Only set a fixed height if there are enough messages
        if chat_history_length >= CHAT_HEIGHT_THRESHOLD:
            chat_container = st.container(height=CHAT_HEIGHT_PX)
        else:
            chat_container = st.container()
            
        with chat_container:
            if chat_history_length == 0:
                st.info("Ask a question to start the conversation!")
            else:
                # Display chat history
                for message in st.session_state.chat_history:
                    role, content = message
                    if role == "human":
                        st.write(user_template.replace("{{MSG}}", content), unsafe_allow_html=True)
                    else:
                        st.write(bot_template.replace("{{MSG}}", content), unsafe_allow_html=True)
        
        # User input section
        st.markdown("---")
        
        # Define a callback for the send button
        def send_message():
            if st.session_state.user_input:
                user_question = st.session_state.user_input
                # Clear input before processing to avoid the error
                st.session_state.user_input = ""
                handle_user_input(user_question)
        
        # Text input and send button
        st.text_input("Ask a question about your documents:", key="user_input", on_change=send_message)
        
        if st.button("Send", on_click=send_message):
            pass  # The actual processing happens in the callback

if __name__=="__main__":
    main()