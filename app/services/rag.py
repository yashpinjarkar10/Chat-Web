from __future__ import annotations

from functools import lru_cache

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_mistralai import MistralAIEmbeddings

from app.core.config import settings


@lru_cache(maxsize=1)
def get_rag_chain():
    if not settings.groq_api_key:
        raise RuntimeError("Missing GROQ_API_KEY in environment")
    if not settings.mistral_api_key:
        raise RuntimeError("Missing MistralAI in environment")

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=settings.mistral_api_key)

    db = Chroma(persist_directory=str(settings.chroma_persist_dir), embedding_function=embeddings)
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 5})

    llm = ChatGroq(model="llama-3.1-8b-instant", api_key=settings.groq_api_key)

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

    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

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

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)
