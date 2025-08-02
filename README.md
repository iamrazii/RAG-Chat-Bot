RAG PDF Chatbot
This is a Streamlit-based Retrieval-Augmented Generation (RAG) chatbot that allows users to upload their own PDF files and ask questions about the content. It uses LangChain, Google Generative AI (Gemini), and a hybrid FAISS + BM25 retriever setup for context-aware answers.

Features

  📝 Upload your own PDF document
  🤖 Ask multi-turn questions in natural language
  📎 Maintains conversational history
  🔍 Hybrid retrieval using semantic (FAISS) + keyword (BM25)
  🧠 Uses intent classification to route:
        Summary questions
        Structural understanding
        Factual lookups
        General information
        Opinions

Tech Stack

LangChain (text splitting, chaining, routing)
Streamlit (frontend)
Google Generative AI (LLM & embeddings)
FAISS Vector DB (semantic search)
BM25 (keyword search)


Installation (Local)

Clone the repo:
  git clone https://github.com/your-username/your-repo-name.git

Install dependencies:
  pip install -r requirements.txt

Set your .env file:
  GOOGLE_TOKEN=your_google_api_key_here
  (or whatever llm you are using)

Run the app:
  streamlit run app.py

Note:

This chatbot may not accurately interpret implicit or unstated information from the uploaded document.

