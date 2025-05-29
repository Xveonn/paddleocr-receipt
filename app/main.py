from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from app.receipt_processor import ReceiptProcessor
import shutil
import os

app = FastAPI()
processor = ReceiptProcessor()

@app.post("/process_receipt/")
async def process_receipt(file: UploadFile = File(...)):
    # Simpan gambar sementara
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Jalankan OCR
    result = processor.process_receipt(temp_path)

    # Hapus file sementara
    os.remove(temp_path)

    return JSONResponse(content=result)

@app.get("/")
def read_root():
    return {"message": "Receipt OCR API is running."}
