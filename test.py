from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", api_key=GOOGLE_API_KEY)

# Embed a single text
result = embeddings.embed_query("Hello, world!")
print(len(result), result[:10])  # print length and first 10 values
