from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import shutil
import os
from embed import build_index
from rag import generate_answer, get_available_documents, generate_risk_flags

app = FastAPI()

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    import tempfile
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(contents)
        tmp_path = tmp.name
    build_index(tmp_path)
    os.remove(tmp_path)
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