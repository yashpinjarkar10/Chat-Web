import time

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain_mistralai import MistralAIEmbeddings

from app.core.config import settings


def build_vectorstore() -> None:
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

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

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
