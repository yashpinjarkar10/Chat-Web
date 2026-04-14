import getpass
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_community.document_loaders import TextLoader
import asyncio

load_dotenv()

# Environment variables with error checking
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

# Check if all required environment variables are present
required_vars = {
    "GROQ_API_KEY": GROQ_API_KEY,
    "NEO4J_URI": NEO4J_URI,
    "NEO4J_USERNAME": NEO4J_USERNAME,
    "NEO4J_PASSWORD": NEO4J_PASSWORD
}

for var_name, var_value in required_vars.items():
    if not var_value:
        raise ValueError(f"Missing required environment variable: {var_name}")

llm = ChatGroq(model="llama-3.1-8b-instant", api_key=GROQ_API_KEY, temperature=0)

current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "information.txt")

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Information file not found: {file_path}")

graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="yashisgood",
    enhanced_schema=True,
)

from langchain_core.documents import Document

loader = TextLoader(file_path)
documents = loader.load()

llm_transformer_filtered = LLMGraphTransformer(
    llm=llm,
)

async def main():
    # Use await with the async function
    graph_documents_filtered = await llm_transformer_filtered.aconvert_to_graph_documents(
        documents
    )
    
    # Add the graph documents to Neo4j
    graph.add_graph_documents(graph_documents_filtered, baseEntityLabel=True)
    print(f"Nodes:{graph_documents_filtered[0].nodes}")
    print(f"Relationships:{graph_documents_filtered[0].relationships}")
    print("Graph documents successfully added to Neo4j!")

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())
    