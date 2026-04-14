import time

import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings

from app.core.config import settings


def _hard_split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be >= 0")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be < chunk_size")

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


def _ensure_max_chunk_size(
    docs: list[Document], *, chunk_size: int, chunk_overlap: int
) -> list[Document]:
    fixed: list[Document] = []
    for doc in docs:
        text = doc.page_content
        if len(text) <= chunk_size:
            fixed.append(doc)
            continue

        for idx, chunk in enumerate(
            _hard_split_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ):
            fixed.append(
                Document(
                    page_content=chunk,
                    metadata={
                        **(doc.metadata or {}),
                        "hard_split": True,
                        "hard_split_index": idx,
                    },
                )
            )
    return fixed


def build_vectorstore() -> None:
    # Windows: avoid noisy HuggingFace cache symlink warnings unless the user opted-in.
    if os.name == "nt" and os.getenv("HF_HUB_DISABLE_SYMLINKS_WARNING") is None:
        os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

    persist_dir = settings.chroma_persist_dir
    info_file = settings.information_file

    if persist_dir.exists():
        print("Vector store already exists. No need to initialize.")
        return

    print("Persistent directory does not exist. Initializing vector store...")

    if not info_file.exists():
        raise FileNotFoundError(
            f"The file {info_file} does not exist. Please check the path (INFORMATION_FILE)."
        )

    loader = TextLoader(str(info_file))
    documents = loader.load()

    chunk_size = 1000
    chunk_overlap = 100
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n## ",
            "\n### ",
            "\n#### ",
            "\n- ",
            "\n\n",
            "\n",
            " ",
            "",
        ],
    )
    docs = text_splitter.split_documents(documents)
    docs = _ensure_max_chunk_size(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    print("\n--- Creating embeddings ---")
    if not settings.mistral_api_key:
        raise RuntimeError("Missing MistralAI in environment")

    embeddings = MistralAIEmbeddings(model="mistral-embed", api_key=settings.mistral_api_key)
    print("\n--- Finished creating embeddings ---")

    print("\n--- Creating vector store ---")
    max_retries = 3
    retry_delay = 20

    for attempt in range(max_retries):
        try:
            Chroma.from_documents(docs, embeddings, persist_directory=str(persist_dir))
            print("\n--- Finished creating vector store ---")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(
                    f"\n--- Error creating vector store: {e}. "
                    f"Retrying in {retry_delay}s ({attempt + 2}/{max_retries}) ---"
                )
                time.sleep(retry_delay)
                retry_delay *= 2
            else:
                raise


if __name__ == "__main__":
    build_vectorstore()
