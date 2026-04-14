title: Webchat1
emoji: 🦀
colorFrom: green
colorTo: pink
sdk: docker
pinned: false
# FastAPI Chatbot

This project is a FastAPI-based chatbot with Retrieval-Augmented Generation (RAG) capabilities, deployed on Hugging Face Spaces. The chatbot uses Groq for chat completion, Mistral for embeddings, and a Chroma vector database to provide contextual responses based on chat history and retrieved documents.

## Deployment Link
[FastAPI Chatbot on Hugging Face Spaces](https://yashpinjarkar10-webchat1.hf.space)

## Features
- Uses Groq for chat responses.
- Uses Mistral for embeddings.
- Implements Retrieval-Augmented Generation (RAG) for better contextual responses.
- Stores and retrieves chat history for improved conversations.
- Includes history-aware retrieval to refine user queries.
- Deployed on Hugging Face Spaces.

## Technologies Used
- **FastAPI**: Web framework for the chatbot API.
- **LangChain**: Framework for building LLM-based applications.
- **Groq**: Provides chat completions.
- **Mistral**: Provides embeddings.
- **Chroma**: Vector database for document storage and retrieval.
- **Uvicorn**: ASGI server for running FastAPI.

## Installation & Setup
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables in a `.env` file:
   ```
  GROQ_API_KEY=your_groq_api_key
  MistralAI=your_mistral_api_key
   ```
4. Run the FastAPI server:
   ```bash
  uvicorn app.main:app --host 0.0.0.0 --port 8080
   ```
5. Access the API at `http://localhost:8080`.

## Project Structure
```
app/
  main.py              # FastAPI app factory + middleware
  api/routes/chat.py   # API routes
  services/rag.py      # RAG chain setup (Groq + Chroma + Mistral embeddings)
  core/config.py       # env + paths
scripts/
  build_vectorstore.py # builds the Chroma store from information.txt
db100/                 # persisted Chroma DB (existing)
information.txt        # source data (you can move this to data/information.txt)
```

## API Endpoints
### Home Route
- **GET `/`**: Check if the FastAPI server is running.

### Start Chat
- **POST `/start`**: Resets chat history and starts a new session.

### Chat Interaction
- **POST `/chat`**: Sends a user query and receives a response.
  - Request Body:
    ```json
    {
      "input": "Hello, how are you?"
    }
    ```
  - Response:
    ```json
    {
      "answer": "I'm doing well! How can I assist you?"
    }
    ```
  - If "exit" is sent as input, it returns an error and asks the user to reset the session.

## Deployment
The chatbot is deployed on Hugging Face Spaces, allowing easy access without local setup.

## License
This project is open-source and available for modification and enhancement.

