import os ,re, fitz
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from dotenv import load_dotenv

load_dotenv()
def extract_pdf_text_pymupdf(path):
    doc = fitz.open(path)
    texts = []
    for i, page in enumerate(doc):
        text = page.get_text("text")
        texts.append(text)
    return "\n\n".join(texts)

def normalize_math_text(t: str) -> str:
    t = t.replace("-", "-").replace("-", "-").replace("—", "-")
    t = t.replace("", "-").replace("", "x")
    t = re.sub(r'(?<=x)(\d)', r'^\1', t)   # crude x2 → x^2
    t = re.sub(r'(\S)-\n(\S)', r'\1\2', t)
    t = re.sub(r'[ \t]*\n[ \t]*(?=\w|\()', ' ', t)
    t = re.sub(r'[ \t]{2,}', ' ', t)
    t = re.sub(r'\n?\s*\d+\s+MATHEMATICS\s*\n?', '\n', t, flags=re.I)
    return t.strip()

pdf_folder = r"D:\Coding\Math-Proffesor-Agent\jemh1dd"   

pdf_docs = []
for fn in os.listdir(pdf_folder):
    if fn.lower().endswith(".pdf"):
        full = os.path.join(pdf_folder, fn)
        txt = extract_pdf_text_pymupdf(full)
        txt = normalize_math_text(txt)
        # ✅ Convert to Document
        pdf_docs.append(Document(page_content=txt, metadata={"source": full}))

# --- Web pages ---
urls = [
    "https://www.vedantu.com/cbse/important-questions-class-10-maths"
]
web_docs = []
for url in urls:
    loader = WebBaseLoader(url)
    web_docs.extend(loader.load())  

# --- Combine everything ---
all_docs = pdf_docs + web_docs

# --- Split into chunks ---
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    separators=["\n\nExample", "\n\nEXAMPLE", "\n\n", "\n", " "]
)

docs = splitter.split_documents(all_docs)

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# FAISS DB
vectorstore = FAISS.from_documents(docs, embedding_model)

# Save FAISS DB locally
vectorstore.save_local("faiss_math_kb")

new_vectorstore = FAISS.load_local("faiss_math_kb", embedding_model, allow_dangerous_deserialization=True)

