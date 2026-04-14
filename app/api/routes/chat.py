from fastapi import APIRouter, HTTPException
from langchain_core.messages import AIMessage, HumanMessage

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag import get_rag_chain

router = APIRouter()

# Global chat history (shared across all clients/processes)
chat_history = []


@router.get("/")
async def root():
    return {"message": "FastAPI Server is Running!"}


@router.post("/start")
async def start_chat():
    global chat_history
    chat_history = []
    return {"message": "Chat session started. Chat history has been reset."}


@router.post("/chat", response_model=ChatResponse)
async def chat(chat_request: ChatRequest):
    global chat_history

    query = chat_request.input
    if query.lower() == "exit":
        raise HTTPException(status_code=400, detail="Use /start to reset the chat session.")

    filtered_chat_history = [
        msg for msg in chat_history if isinstance(msg, HumanMessage) or isinstance(msg, AIMessage)
    ]

    rag_chain = get_rag_chain()
    result = rag_chain.invoke({"input": query, "chat_history": filtered_chat_history})

    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result["answer"]))

    return ChatResponse(answer=result["answer"])
