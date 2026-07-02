from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from embed import build_index
from rag import generate_answer, get_available_documents, generate_risk_flags

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

UPLOAD_DIR = "uploads"

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
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
async def ask_question(request: QueryRequest):
    answer = generate_answer(request.question)
    return {"answer": answer}

@app.get("/risks")
async def get_risk_flags():
    flags = generate_risk_flags()
    return {"risks": flags}
