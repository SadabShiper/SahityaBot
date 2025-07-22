import re
import os
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
import nltk
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

nltk.download("punkt")

# === CONFIG ===
GEMINI_API_KEY = "AIzaSyBlC2U2XnfmwfsHglFcxlISJVdulq6FvXM"
genai.configure(api_key=GEMINI_API_KEY)

MODEL_NAME = "models/gemini-2.5-pro"
SHORT_TERM_LIMIT = 10

# === PREPROCESS ===
def clean_text(text):
    text = re.sub(r'[A-Za-z0-9~!@#$%^&*()_+=\[\]{}|\\:;"\'<>,./?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_sentences(text):
    from nltk.tokenize import sent_tokenize
    return sent_tokenize(text, language='english')

def preprocess_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    cleaned = clean_text(raw_text)
    chunks = split_into_sentences(cleaned)
    return chunks

print("📄 Preprocessing corpus...")
chunks = preprocess_file("output_bangla_pytesseract_300.txt")
print(f"✅ Got {len(chunks)} chunks.")

# === FAISS ===
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

print("🔍 Embedding chunks...")
embeddings = np.array([embedder.encode(chunk) for chunk in tqdm(chunks)], dtype='float32')

dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)

print(f"✅ FAISS index built with {index.ntotal} vectors.")

# === CHATBOT LOGIC ===
def query_faiss(question, top_k=3):
    query_emb = np.array([embedder.encode(question)], dtype='float32')
    distances, indices = index.search(query_emb, top_k)
    return [chunks[idx] for idx in indices[0]]

def build_chat_prompt(history, kb_context, question):
    chat_history_str = ""
    for turn in history:
        chat_history_str += f"প্রশ্ন: {turn['user']}\nউত্তর: {turn['bot']}\n"
    chat_history_str += f"প্রশ্ন: {question}\n"
    
    prompt = f"""তুমি একজন সহায়ক বাংলা চ্যাটবট। নিচে পূর্ববর্তী কথোপকথন এবং প্রাসঙ্গিক তথ্য দেয়া হয়েছে। এগুলোর ভিত্তিতে সংক্ষিপ্ত এবং সঠিক উত্তর দাও।
    
    প্রাসঙ্গিক তথ্য: {kb_context}

    কথোপকথন:
    {chat_history_str}
    উত্তর:"""
    return prompt

def generate_answer(prompt):
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(prompt)
    return response.text.strip()

# === FASTAPI ===
app = FastAPI(title="Bangla RAG Chatbot")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Session store
sessions = {}

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id
    message = request.message.strip()
    
    # Get/create session history
    if session_id not in sessions:
        sessions[session_id] = []
    history = sessions[session_id]
    
    # Retrieve KB context
    retrieved = query_faiss(message)
    context = " ".join(retrieved)
    
    # Build prompt and get answer
    prompt = build_chat_prompt(history, context, message)
    answer = generate_answer(prompt)
    
    # Save turn to history
    history.append({"user": message, "bot": answer})
    if len(history) > SHORT_TERM_LIMIT:
        history.pop(0)
    
    return {
        "session_id": session_id,
        "user_message": message,
        "bot_reply": answer,
        "chat_history": history,
        "retrieved_contexts": retrieved
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("rag_chatbot:app", host="0.0.0.0", port=port)