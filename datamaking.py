import os
import time
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai._common import GoogleGenerativeAIError
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MistrakAI_API_KEY = os.getenv("MistralAI")
# Define the directory containing the text file and the persistent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "information.txt")
persistent_directory = os.path.join(current_dir, "db100", "chroma_db100")

# Check if the Chroma vector store already exists
if not os.path.exists(persistent_directory):
    print("Persistent directory does not exist. Initializing vector store...")

    # Ensure the text file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"The file {file_path} does not exist. Please check the path."
        )

    # Read the text content from the file
    loader = TextLoader(file_path)
    documents = loader.load()

    # Split the document into chunks (larger chunks to reduce API calls)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)

    # Display information about the split documents
    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")
    print(f"Sample chunk:\n{docs[0].page_content}\n")

    # Create embeddings with batch processing configuration
    print("\n--- Creating embeddings ---")
    
    # Configure embeddings with batch processing to reduce API calls
    # embeddings = GoogleGenerativeAIEmbeddings(
    #     model="models/embedding-001", 
    #     api_key=GOOGLE_API_KEY,
    #     # Add any batch configuration if available
    # )


    embeddings = MistralAIEmbeddings(
        model="mistral-embed",
        api_key=MistrakAI_API_KEY
    )
    print("\n--- Finished creating embeddings ---")

    # Create the vector store and persist it automatically with error handling
    print("\n--- Creating vector store ---")
    max_retries = 3
    retry_delay = 20  # Wait 60 seconds between retries
    
    for attempt in range(max_retries):
        try:
            db = Chroma.from_documents(
                docs, embeddings, persist_directory=persistent_directory)
            print("\n--- Finished creating vector store ---")
            break
        except GoogleGenerativeAIError as e:
            if "429" in str(e) or "quota" in str(e).lower():
                if attempt < max_retries - 1:
                    print(f"\n--- API Quota exceeded. Waiting {retry_delay} seconds before retry {attempt + 2}/{max_retries} ---")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print("\n--- Max retries reached. Please check your API quota and billing details ---")
                    print("Consider using a local embedding model or increasing your API quota.")
                    raise
            else:
                print(f"\n--- Unexpected Google API error: {e} ---")
                raise
        except Exception as e:
            print(f"\n--- Unexpected error: {e} ---")
            raise

else:
    print("Vector store already exists. No need to initialize.")