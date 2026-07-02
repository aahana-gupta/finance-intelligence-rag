from fastapi import FastAPI, UploadFile, File, Request
from pydantic import BaseModel
import shutil
import os
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from embed import build_index
from rag import generate_answer, get_available_documents, generate_risk_flags

limiter = Limiter(key_func=get_remote_address)
app = FastAPI()
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

class QueryRequest(BaseModel):
    question: str

UPLOAD_DIR = "uploads"

@app.post("/upload")
@limiter.limit("10/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    safe_name = os.path.basename(file.filename or "")
    if not safe_name.lower().endswith(".pdf"):
        return {"error": "Only PDF files are allowed"}
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    pdf_path = os.path.join(UPLOAD_DIR, safe_name)
    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    build_index(pdf_path)
    return {"message": "PDF uploaded and indexed successfully"}

@app.get("/documents")
async def list_documents():
    return {"documents": get_available_documents()}

@app.post("/ask")
@limiter.limit("10/minute")
async def ask_question(request: Request, query: QueryRequest):
    answer = generate_answer(query.question)
    return {"answer": answer}

@app.get("/risks")
async def get_risk_flags():
    flags = generate_risk_flags()
    return {"risks": flags}
