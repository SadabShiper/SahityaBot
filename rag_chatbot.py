# rag_chatbot_cleaned.py

from time import sleep
import re
import os
import json
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder
import google.generativeai as genai
import nltk
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from dotenv import load_dotenv
from uuid import uuid4
import faiss
from typing import List
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from evaluation_report import RAGEvaluator

# === INIT ===
load_dotenv()
nltk.download("punkt")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)
MODEL_NAME = "models/gemini-2.5-flash"
TEXT_FILE = "output_bangla_pytesseract_300.txt"
SHORT_TERM_LIMIT = 10
TOP_K = 5
RELEVANCE_THRESHOLD = 0.4

# === PREPROCESS ===
def clean_text(text):
    text = re.sub(r'[A-Za-z0-9~!@#$%^&*()_+=\[\]{}|\\:;"\'<>,./?-]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def split_into_chunks(text, chunk_size=3, stride=1):
    sents = nltk.sent_tokenize(text, language='english')
    return [" ".join(sents[i:i+chunk_size]) for i in range(0, len(sents), stride) if sents[i:i+chunk_size]]

def preprocess_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    return split_into_chunks(clean_text(raw_text))

print("📄 Preprocessing corpus...")
chunks = preprocess_file(TEXT_FILE)
print(f"✅ Got {len(chunks)} chunks.")

# === FAISS ===
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)
print("🔍 Embedding chunks...")
embeddings = np.array([embedder.encode(c, normalize_embeddings=True) for c in tqdm(chunks)], dtype='float32')
dim = embeddings.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings)
print(f"✅ FAISS index built with {index.ntotal} vectors.")

cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", local_files_only=True)
evaluator = RAGEvaluator()

# === RAG Core ===
def query_faiss(question, top_k=TOP_K):
    query_emb = np.array([embedder.encode(question, normalize_embeddings=True)], dtype='float32')
    distances, indices = index.search(query_emb, top_k)
    retrieved = [chunks[idx] for idx in indices[0]]
    scores = cross_encoder.predict([[question, c] for c in retrieved])
    return [c for c, _ in sorted(zip(retrieved, scores), key=lambda x: -x[1])]

def build_chat_prompt(history, kb_context, question):
    chat_history_str = "".join([f"প্রশ্ন: {turn['user']}\nউত্তর: {turn['bot']}\n" for turn in history])
    chat_history_str += f"প্রশ্ন: {question}\n"
    return f"""তুমি একজন সহায়ক বাংলা চ্যাটবট। নিচে পূর্ববর্তী কথোপকথন এবং প্রাসঙ্গিক তথ্য দেয়া হয়েছে। এগুলোর ভিত্তিতে সংক্ষিপ্ত এবং সঠিক উত্তর দাও। অনুমান করো না। যদি প্রাসঙ্গিক তথ্যের মধ্যে উত্তর না থাকে, সরাসরি বলো যে উত্তর পাওয়া যায়নি।

প্রাসঙ্গিক তথ্য:
{kb_context}

কথোপকথন:
{chat_history_str}
উত্তর:"""

def generate_answer(prompt):
    try:
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Generation error: {str(e)}")
        if "429" in str(e) or "quota" in str(e).lower():
            return "[RATE_LIMIT_ERROR] API quota exceeded. Please try again later."
        return "[GENERATION_ERROR] দুঃখিত একটি ত্রুটি ঘটেছে, অনুগ্রহ করে একটু পরে আবার চেষ্টা করুন!"

def groundedness_score(answer, context_chunks):
    ans_emb = embedder.encode(answer, normalize_embeddings=True)
    ctx_embs = [embedder.encode(c, normalize_embeddings=True) for c in context_chunks]
    return float(np.max(cosine_similarity([ans_emb], ctx_embs)[0]))

# === FASTAPI ===
app = FastAPI(title="Bangla RAG Chatbot")
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")
sessions = {}

class ChatRequest(BaseModel):
    session_id: str
    message: str

@app.get("/")
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/chat")
async def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid4())
    message = request.message.strip()
    sessions.setdefault(session_id, [])
    history = sessions[session_id]

    try:
        retrieved = query_faiss(message)
        context = " |SEP| ".join(retrieved)
        prompt = build_chat_prompt(history, context, message)
        full_answer = generate_answer(prompt)

        if full_answer.startswith("[RATE_LIMIT_ERROR]"):
            return StreamingResponse(iter([full_answer.replace("[RATE_LIMIT_ERROR]", "")]), media_type="text/plain")
        if full_answer.startswith("[GENERATION_ERROR]"):
            return StreamingResponse(iter([full_answer.replace("[GENERATION_ERROR]", "")]), media_type="text/plain")

        grounding = groundedness_score(full_answer, retrieved)
        if grounding < RELEVANCE_THRESHOLD:
            full_answer = "দুঃখিত, প্রাসঙ্গিক তথ্যের মধ্যে উত্তর পাওয়া যায়নি।"

        history.append({"user": message, "bot": ""})
        if len(history) > SHORT_TERM_LIMIT:
            history.pop(0)

        evaluator.log_interaction(session_id, message, context, full_answer)

        def stream():
            partial = ""
            for c in full_answer:
                partial += c
                yield c
                sleep(0.02)
            history[-1]["bot"] = partial

        return StreamingResponse(stream(), media_type="text/plain")

    except Exception as e:
        print(f"Chat error: {str(e)}")
        msg = "দুঃখিত একটি ত্রুটি ঘটেছে, অনুগ্রহ করে একটু পরে আবার চেষ্টা করুন!"
        return StreamingResponse(iter([msg]), media_type="text/plain")

@app.get("/api/sessions")
async def get_sessions():
    return {"sessions": list(sessions.keys())}

@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    return {"history": sessions.get(session_id, [])}

@app.get("/api/generate_report")
async def generate_report():
    try:
        markdown = evaluator.generate_report("markdown")
        html = evaluator.generate_report("html")
        with open("evaluation_report.md", "w", encoding="utf-8") as f:
            f.write(markdown)
        with open("evaluation_report.html", "w", encoding="utf-8") as f:
            f.write(html)
        return {"status": "success", "files": ["evaluation_report.md", "evaluation_report.html"]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/evaluation_report")
async def get_evaluation_report(format: str = "json"):
    df, summary = evaluator.generate_evaluation_table()
    if format.lower() == "html":
        html = evaluator.generate_report("html")
        return HTMLResponse(content=html)
    return {"summary": summary, "detailed_logs": df.to_dict("records")}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("rag_chatbot_cleaned:app", host="0.0.0.0", port=port)
