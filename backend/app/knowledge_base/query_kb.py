from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

import os 
from dotenv import load_dotenv

load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")



embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Load FAISS index
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent   
VECTORSTORE_PATH = BASE_DIR / "faiss_math_kb"


vectorstore = FAISS.load_local(
    VECTORSTORE_PATH, 
    embedding_model, 
    allow_dangerous_deserialization=True
)

# LLM setup
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.2
)

# Prompt template
prompt_template = """
You are an expert math tutor. 
Answer the question based ONLY on the provided context. 
If context is insufficient, say "Not enough information in knowledge base".

Question:
{question}

Context:
{context}

Answer:
"""
prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=prompt_template,
)

chain = LLMChain(llm=llm, prompt=prompt)

def query_kbf(query: str) -> str:
    """Query the knowledge base and return only the answer string."""
    
    context_docs = vectorstore.similarity_search(query, k=2)
    context = "\n".join([doc.page_content for doc in context_docs])
    
    # Run LLM chain
    answer = chain.run({"question": query, "context": context})
    return answer

