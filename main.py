# main.py - Updated with FAISS, MMR, and Dynamic Prompts

import os
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import FAISS # MODIFIED: Using FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# Using robust text extraction for Bangla
from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import fitz  # PyMuPDF for manual text extraction
import unicodedata

# --- CONFIGURATION & CONSTANTS ---
load_dotenv()

# Load Google API Key from environment variables
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") # Added default for convenience
if not GOOGLE_API_KEY:
    raise ValueError("Google API Key not found. Please set it as an environment variable.")

PDF_FILE_PATH = "HSC26_Bangla_1st_paper.pdf"
VECTOR_DB_PATH = "faiss_index_bangla" # MODIFIED: Path for FAISS index
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
LLM_MODEL_NAME = "gemini-2.5-flash"

# MODIFIED: Using the dynamic prompt template from the notebook
BASE_PROMPT_TEMPLATE = """You are a helpful AI assistant for answering questions about a given document.
You are given a question with desired language you should answer to and a set of document chunks as context IN BANGLA LANGUAGE.
You must STRICTLY FOLLOW these rules:
1. EVEN THOUGH THE CONTEXT WILL BE IN BANGLA, YOU SHOULD ALWAYS ANSWER IN THE DESIRED LANGUAGE: {language_placeholder}.
2. If the information to answer the question is not in the context, you MUST respond with one of the following sentences, matching the desired language:
   - For English questions: "Sorry, I am unable to answer this question."
   - For Bangla questions: "দুঃখিত, আপনার প্রশ্নটির উত্তর আমার জানা নেই।"
3. Try to answer only using information present in the context and your reasoning capability.
Context:
{context}

Question: {question}
Desired Answer Language: {language_placeholder}
Answer:"""

CONDENSE_QUESTION_PROMPT_TEMPLATE = """Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question. The standalone question must be in the same language as the follow-up question.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone question:"""

CONDENSE_PROMPT = PromptTemplate.from_template(CONDENSE_QUESTION_PROMPT_TEMPLATE)

# --- FASTAPI APP INITIALIZATION ---
app = FastAPI(
    title="Multilingual RAG Chatbot API",
    description="An API for a RAG chatbot that answers questions about a document in English and Bengali.",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.state.chain = None

# --- HELPER FUNCTIONS ---

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': False}
hf = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)

from pdf2image import convert_from_path
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

def ocr_extract_bangla_text(pdf_path):
    images = convert_from_path(pdf_path)
    text_list = []
    for img in images:
        text = pytesseract.image_to_string(img, lang='ben')
        text_list.append(text)
    return "\n\n".join(text_list)

def detect_language(text):
    bangla_count = sum(0x0980 <= ord(char) <= 0x09FF for char in text)
    english_count = sum(0x0041 <= ord(char) <= 0x007A for char in text)
    return "Bangla" if bangla_count > english_count else "English"

# MODIFIED: This function now loads or creates a FAISS index
def load_and_embed_pdf():
    if os.path.exists(VECTOR_DB_PATH):
        print("Loading existing FAISS vector database...")
        return FAISS.load_local(
            VECTOR_DB_PATH,
            embeddings=hf,
            allow_dangerous_deserialization=True # Required for loading
        )

    if not os.path.exists(PDF_FILE_PATH):
        raise FileNotFoundError(f"PDF file not found at {PDF_FILE_PATH}. Please make sure it's in the same directory.")

    print(f"Loading and processing '{PDF_FILE_PATH}' with robust extractor...")
    documents = extract_text_with_pymupdf(PDF_FILE_PATH)

    if not documents:
        raise ValueError("Could not extract any text from the PDF. Please check the file.")
    
    text_lists = ocr_extract_bangla_text(PDF_FILE_PATH)

    from langchain.schema import Document
    documents = [Document(page_content=text_lists, metadata={"source": "ocr_bangla"})] 

    for doc in documents:
        doc.page_content = doc.page_content.replace('\n', ' ').strip()

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", "।", "?", " ","!"]  # Add Bangla sentence marker
    )
    chunks = text_splitter.split_documents(documents)

    print(f"Creating FAISS vector embeddings for {len(chunks)} chunks...")
    vector_store = FAISS.from_documents(documents=chunks, embedding=hf)
    vector_store.save_local(VECTOR_DB_PATH)
    print(f"FAISS index created and saved successfully at '{VECTOR_DB_PATH}'!")
    return vector_store

# ADDED: Helper function to create language-specific prompts dynamically
def create_language_specific_prompt(language):
    """Create a prompt template with the language substituted."""
    template = BASE_PROMPT_TEMPLATE.replace("{language_placeholder}", language)
    return PromptTemplate(template=template, input_variables=["context", "question"])

# MODIFIED: Initializes the chain with MMR retriever and a default prompt
def initialize_llm_and_chain(vector_store):
    if vector_store is None:
        return None

    llm = ChatGoogleGenerativeAI(
        model=LLM_MODEL_NAME,
        google_api_key=GOOGLE_API_KEY,
        temperature=0.7,
        convert_system_message_to_human=True
    )

    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True, output_key='answer'
    )

    # MODIFIED: Using MMR for retrieval to get more diverse results
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={'k': 10, 'lambda_mult': 0.7}
    )

    # Create a default prompt. It will be updated for each request in the /chat endpoint.
    default_prompt = create_language_specific_prompt("English")

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        condense_question_prompt=CONDENSE_PROMPT,
        return_source_documents=True,
        verbose=True,
        combine_docs_chain_kwargs={"prompt": default_prompt}
    )
    return chain

# --- STARTUP EVENT ---
@app.on_event("startup")
def on_startup():
    print("Application startup: Initializing services...")
    try:
        vector_store = load_and_embed_pdf()
        app.state.chain = initialize_llm_and_chain(vector_store)
        if app.state.chain:
            print("✅ RAG chain initialized successfully.")
        else:
            print("❌ Error: RAG chain could not be initialized.")
    except Exception as e:
        print(f"❌ An error occurred during startup initialization: {e}")
        app.state.chain = None

# --- API & FRONTEND ENDPOINTS ---

@app.get("/", response_class=FileResponse, tags=["Frontend"])
async def read_index():
    if not os.path.exists("index.html"):
        raise HTTPException(status_code=404, detail="index.html not found")
    return "index.html"

@app.get("/api/status", tags=["Status"])
def get_status():
    chain_status = "initialized" if app.state.chain else "not initialized"
    return {"status": "ok", "message": "Welcome!", "chain_status": chain_status}

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str
    source_documents: list

# MODIFIED: Chat endpoint now dynamically updates the prompt
@app.post("/api/chat", response_model=ChatResponse, tags=["Chat"])
async def chat_with_document(request: ChatRequest):
    if not app.state.chain:
        raise HTTPException(status_code=503, detail="RAG chain is not initialized.")
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        # 1. Detect language from the user's query
        language = detect_language(request.query)
        print(f"Detected language: {language}")

        # 2. Create a new prompt template with the detected language
        language_specific_prompt = create_language_specific_prompt(language)

        # 3. Update the chain's prompt template for this specific request
        app.state.chain.combine_docs_chain.llm_chain.prompt = language_specific_prompt

        # 4. Call the chain with the original question
        result = await app.state.chain.ainvoke({"question": request.query})

        source_docs_formatted = [
            {"page_content": doc.page_content, "metadata": doc.metadata}
            for doc in result.get('source_documents', [])
        ]

        return ChatResponse(
            answer=result.get('answer', 'No answer found.'),
            source_documents=source_docs_formatted
        )
    except Exception as e:
        print(f"An error occurred during chat processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print("Starting Uvicorn server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)