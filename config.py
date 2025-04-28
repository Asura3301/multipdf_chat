"""
Configuration settings for the MultiPDF Chat application.
"""

# Application settings
APP_TITLE = "MultiPDF Chat Assistant"
APP_ICON = "📚"
APP_LOGO = "https://avatars.githubusercontent.com/u/115102523?v=4"
LOGO_WIDTH = 70

# PDF processing settings
DEFAULT_CHUNK_SIZE = 512
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 1000
CHUNK_SIZE_STEP = 50

DEFAULT_CHUNK_OVERLAP = 64
MIN_CHUNK_OVERLAP = 0
MAX_CHUNK_OVERLAP = 200
CHUNK_OVERLAP_STEP = 10

# LLM settings
DEFAULT_MODEL = "llama-3.1-8b-instant"
ALTERNATIVE_MODELS = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
    "deepseek-r1-distill-llama-70b-specdec"
]
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 1024

# Embedding settings
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Vector store settings
RETRIEVER_SEARCH_TYPE = "similarity"
RETRIEVER_K = 3

# UI settings
CHAT_HEIGHT_THRESHOLD = 4  # Number of messages before scrolling
CHAT_HEIGHT_PX = 500  # Height in pixels when scrolling is enabled

# Sample questions
SAMPLE_QUESTIONS = [
    "What are the main topics covered in these documents?",
    "Can you summarize the key points from all documents?",
    "What are the relationships between concepts in these documents?"
]

# About text
ABOUT_TEXT = """
This app allows you to chat with multiple PDF documents using LangChain and Groq LLM.

Built with Groq, Streamlit, LangChain, Google Serper, and FAISS vector database.

GitHub: [Asura3301](https://github.com/Asura3301/multipdf_chat)

Kaggle: [MultiPDF Chat](https://www.kaggle.com/code/adastra3301/multipdf-chat)

Medium: [Gen AI Intensive Course Capstone 2025Q1](https://medium.com/@evezon00/gen-ai-intensive-course-capstone-2025q1-75d2e94ab86b)
"""