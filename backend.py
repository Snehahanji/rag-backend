import os
from typing import List

from fastapi import FastAPI, UploadFile, File
from dotenv import load_dotenv
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# ---------------- LOAD ENV ----------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ["HF_TOKEN"] = HF_TOKEN

# ---------------- FASTAPI APP ----------------

app = FastAPI()

# ---------------- MODELS ----------------

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

llm = ChatGroq(
    groq_api_key=GROQ_API_KEY,
    model_name="llama-3.1-8b-instant"
)

# ---------------- GLOBAL MEMORY ----------------

store = {}
vectorstore = None
conversational_rag = None

# ---------------- CHAT MEMORY ----------------

def get_history(session) -> BaseChatMessageHistory:
    if session not in store:
        store[session] = ChatMessageHistory()
    return store[session]

# ---------------- UPLOAD ENDPOINT ----------------

@app.post("/upload")
async def upload_pdfs(files: List[UploadFile] = File(...)):

    global vectorstore, conversational_rag

    documents = []

    os.makedirs("temp", exist_ok=True)

    for file in files:
        path = f"temp/{file.filename}"

        with open(path, "wb") as f:
            f.write(await file.read())

        loader = PyPDFLoader(path)
        docs = loader.load()
        documents.extend(docs)

    if len(documents) == 0:
        return {"message": "No readable text found in PDFs"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )

    splits = splitter.split_documents(documents)

    if len(splits) == 0:
        return {"message": "No text chunks created"}

    vectorstore = Chroma.from_documents(splits, embeddings)
    retriever = vectorstore.as_retriever()

    # -------- PROMPTS --------

    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite question as standalone."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    history_aware = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer only from context:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware, qa_chain)

    conversational_rag = RunnableWithMessageHistory(
        rag_chain,
        get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer"
    )

    return {"message": "PDFs processed successfully"}

# ---------------- CHAT ENDPOINT ----------------

@app.post("/chat")
async def chat(query: str, session_id: str = "default"):

    if conversational_rag is None:
        return {"answer": "Upload PDFs first."}

    response = conversational_rag.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )

    return {"answer": response["answer"]}

