import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_community.utilities.google_serper import GoogleSerperAPIWrapper
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import Tool
from langchain_groq import ChatGroq
from langchain import hub
import os
import getpass
import io
from config import *  # Import all configuration settings
from html_templates import css, bot_template, user_template
from google_search import GoogleSearchFallbackChain


#------------------------------------ 
# setup environment variables

# Groq API key for LLM
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = getpass.getpass("Enter your Groq API key: ")

# Serper API key for Google Search
if "SERPER_API_KEY" not in os.environ:
    os.environ["SERPER_API_KEY"] = getpass.getpass("Enter your Serper API key: ")

if "HUGGINGFACE_API_TOKEN" in os.environ:
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    
#------------------------------------

def preprocess_pdf_files(pdf_docs):
    """
    Converts Streamlit uploaded files to a list of tuples (filename, file_content_bytes).
    
    Args:
        pdf_docs: List of uploaded PDF files from Streamlit file uploader
        
    Returns:
        List of tuples: [(filename, file_content_bytes), ...]
    """
    pdf_files_content = []
    
    for pdf in pdf_docs:
        if pdf is not None:
            # Get the filename and read the file content as bytes
            filename = pdf.name
            file_content_bytes = pdf.read()
            
            # Add tuple to the list
            pdf_files_content.append((filename, file_content_bytes))
            
            # Reset file pointer for potential future reads
            pdf.seek(0)
    
    return pdf_files_content

#------------------------------------
def get_pdf_text(pdf_files_content):
    """Extract text from multiple PDF documents."""
    text = ""
    # pdf_files_content is expected to be a list of tuples: (filename, file_content_bytes)
    for filename, content_bytes in pdf_files_content:
        try:
            # Use io.BytesIO to read bytes as a file
            pdf_stream = io.BytesIO(content_bytes)
            pdf_reader = PdfReader(pdf_stream)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n\n--- Document: {filename}, Page {page_num+1} ---\n"
                    text += page_text
        except Exception as e:
            st.error(f"Error processing {filename}: {str(e)}")
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

# TODO: add filenames to metadata
def get_vector_store(text_chunks):
    """Create a vector store from text chunks."""
    if not text_chunks:
        st.error("No text chunks provided!")
        return None
    
    embeddings = download_embeddings()
    if embeddings is None:
        st.error("Failed to load embeddings! Vector store will not be created.")
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
# Setup function for Google Search

def setup_serper_search():
    """Initializes the GoogleSerperAPIWrapper tool."""
    try:
        # Check for Serper API key
        if "SERPER_API_KEY" not in os.environ:
            load_dotenv() # Try .env file
            if "SERPER_API_KEY" not in os.environ:
                # only if not found anywhere
                os.environ["SERPER_API_KEY"] = getpass.getpass("Enter your Serper API key: ")

        # Check if key is actually loaded
        serper_key = os.environ.get("SERPER_API_KEY")
        if not serper_key:
             st.error("SERPER_API_KEY not found. Serper search will be unavailable.")
             return None

        # Initialize Google search wrapper
        serper_search = GoogleSerperAPIWrapper(serper_api_key=serper_key)

        # Test the connection 
        try:
            # test_results = serper_search.results("test query") 
            # st.text("Serper API connection test successful")
            # st.text(f"Test Results: {test_results}") # Uncomment for debugging
            pass
        except Exception as conn_error:
            st.error(f"Serper API connection test failed: {conn_error}")
            # Check if it's an auth error specifically
            if "401" in str(conn_error) or "Unauthorized" in str(conn_error):
                 st.error("Authentication failed. Please check your SERPER_API_KEY.")
            st.error("Falling back to documents-only mode.")
            return None

        # Create the tool - Serper(Google search) wrapper's run method returns a string summary
        serper_search_tool = Tool(
            name="serper_search", 
            description="Useful for retrieving fresh web snippets when the PDF corpus lacks information.",
            func=serper_search.run 
        )

        st.text("Serper Search tool initialized successfully.")
        return serper_search_tool
    except Exception as e:
        st.error(f"Failed to initialize Serper Search tool: {e}")
        return None

#------------------------------------


def get_conversation_chain(vectorstore, model_name):
    """Create a conversation chain."""
    
    # Initialize Google Search tool
    serper_search_tool = setup_serper_search() 
    st.text(f"Google Search tool status: {'Available' if serper_search_tool else 'Not available'}")
    
    if not vectorstore:
        st.error("No vector store provided!")
        return None
    
    try:    
        # init LLM
        llm = ChatGroq(
            temperature=DEFAULT_TEMPERATURE, 
            model_name=model_name, 
            max_tokens=DEFAULT_MAX_TOKENS
        )
        
        # retriever
        retriever = vectorstore.as_retriever(
            search_type=RETRIEVER_SEARCH_TYPE, 
            search_kwargs={"k": RETRIEVER_K},
        )

        
        # Another way to use LLMs via HuggingFace Hub
        # llm = HuggingFaceEndpoint(
        #     repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        #     task="text-generation",
        #     max_new_tokens=100,
        #     do_sample=False,
        # )
        
        # Contextualize question prompt
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )
        
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
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
        
        # Few-shot examples for the QA prompt
        few_shot_examples = [
            {
                "question": "What is the effective date of the contract?",
                "context": "Doc‑1 p.3: 'This agreement shall come into force on 1 July 2024.'",
                "answer": "The contract becomes effective on **1 July 2024**. [Doc‑1, p. 3]"
            },
            {
                "question": "What are the key deliverables?",
                "context": "Doc‑2 p.5: 'The vendor shall provide: (a) monthly progress reports, (b) final software implementation, (c) user documentation.'",
                "answer": "The key deliverables include monthly progress reports, final software implementation, and user documentation. [Doc‑2, p. 5]"
            },
            {
                "question": "Who are the project stakeholders?",
                "context": "",
                "answer": "INSUFFICIENT DATA"
            },
            {
                "question": "What were the latest developments in AI regulation?",
                "context": "",
                "answer": "INSUFFICIENT DATA"
            }
        ]
        
        # Create few-shot examples string
        few_shot_str = "\n\nExamples:\n" + "\n".join([
            f"Question: {ex['question']}\nContext: {ex['context']}\nAnswer: {ex['answer']}\n"
            for ex in few_shot_examples
        ])
        
        # Answer question prompt with few-shot examples
        qa_system_prompt = (
            "You are a senior policy analyst. Use ONLY the provided context. "
            "Generate step‑by‑step reasoning internally, then give a "
            "concise answer ≤120 words with inline citations (e.g., [Doc‑3, p. 7]). "
            "If the answer is not present, respond exactly: 'INSUFFICIENT DATA'."
            f"{few_shot_str}\n\n"
            "Now, use the following context to answer the question:\n"
            "{context}"
        )
        
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ])
        
        # Create combine docs chain
        combine_docs_chain = create_stuff_documents_chain(
            llm,
            qa_prompt,
        )
        
        # Create retrieval chain(Main RAG Chain)
        rag_chain = create_retrieval_chain(
            retriever=history_aware_retriever,
            combine_docs_chain=combine_docs_chain,
        )
        
        # Create the enhanced chain with Google Search fallback if tool is available
        if serper_search_tool:
            enhanced_chain = GoogleSearchFallbackChain(rag_chain, serper_search_tool, llm) 
            st.text("Google Search fallback mechanism enabled.")
            return enhanced_chain
        else:
            # If Serper tool failed to initialize, return the basic RAG chain
            st.text("Proceeding without Google Search fallback.")
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
        return"Error: Please process the documents first.", st.session_state.chat_history

    
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
                    # Preprocess the PDF files
                    pdf_files_content = preprocess_pdf_files(pdf_docs)
            
                    # Extract text
                    raw_text = get_pdf_text(pdf_files_content)
                    
                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    
                    # Create vector store
                    vectorstore = get_vector_store(text_chunks)
                    
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore, model_name)
                    
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