from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Union
import os
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Define the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
persistent_directory = os.path.join(current_dir, "db", "chroma_db")

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)

# Load the existing vector store with the embedding function
db = Chroma(persist_directory=persistent_directory, embedding_function=embeddings)

# Create a retriever for querying the vector store
retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

# Initialize the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", api_key=GOOGLE_API_KEY)

# Contextualize question prompt
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
history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

# Answer question prompt
# Update this prompt to reflect your desired behavior (e.g., act as "you")
qa_system_prompt = (
    "You are an assistant that acts as me. Use the following pieces of retrieved context "
    "to answer the question. If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise. Always respond as if you are me."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Create a chain to combine documents for question answering
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

# Create a retrieval chain that combines the history-aware retriever and the question answering chain
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

app = FastAPI()

# Global chat history
chat_history = []

class ChatRequest(BaseModel):
    input: str

class ChatResponse(BaseModel):
    answer: str
# Enable CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Home route to check if FastAPI is running
@app.get("/")
async def root():
    return {"message": "FastAPI Server is Running!"}
@app.post("/start")
async def start_chat():
    global chat_history
    chat_history = []  # Reset chat history
    return {"message": "Chat session started. Chat history has been reset."}

@app.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    global chat_history

    query = chat_request.input

    if query.lower() == "exit":
        raise HTTPException(status_code=400, detail="Use /start to reset the chat session.")

    # Filter out SystemMessage, keeping only HumanMessage and AIMessage
    filtered_chat_history = [
        msg for msg in chat_history if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)
    ]

    # Invoke the RAG chain
    result = rag_chain.invoke({"input": query, "chat_history": filtered_chat_history})

    # Update the chat history
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result['answer']))

    return ChatResponse(answer=result['answer'])

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)