# 📘 Math Professor Agent

An AI-powered **Math Knowledge Assistant** that uses a **multi-agent, agentic RAG (Retrieval-Augmented Generation)** approach to answer mathematics questions.  

The system processes textbooks (PDFs) and curated web resources, embeds them with **Google Gemini embeddings**, and stores them in a **FAISS vector database**.  
Queries are then handled through a **multi-agent pipeline** where each agent has a specific role (guard, retriever, answer generator).

---

## 🚀 Project Overview

This project follows an **agentic multi-agent structure**:

1. **AI Gateway Agent** – filters and validates user inputs/outputs (ensures only math-related queries are processed).  
2. **Retriever Agent (RAG)** – searches the FAISS knowledge base for relevant context from PDFs and web docs.  
3. **Math Professor Agent** – generates structured, step-by-step answers using the retrieved knowledge.  

This **Agentic RAG pipeline** makes the assistant more reliable, modular, and extendable for future improvements.

---

## 🛠 Tech Stack

- **Python 3.10+**
- **LangChain** – agents, document loaders, text splitters
- **LangGraph** – Multi Agent Structure
- **FAISS** – vector database for semantic search
- **FastAPI** - Backend Hosting
- **Streamlit** – Frontend

---

## ⚙️ Setup & Installation

 **Clone the repository**
   ```bash
   git clone https://github.com/your-username/math-professor-agent.git
   cd math-professor-agent
```
 **Create a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
```
 **Install dependencies**
```bash
pip install -r requirements.txt
```
**Create a .env file**
```bash
GOOGLE_API_KEY=your_api_key_here
TAVILY_API_KEY = your_api_key
```
### Author - Dhairya Jain
