from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import base64
from openai import OpenAI
import pandas as pd
import fitz
from PyPDF2 import PdfReader
import openpyxl
import tempfile
from typing import List, Tuple, Dict
import re
import datetime
from dotenv import load_dotenv
import json

load_dotenv()

app = FastAPI(title="PCI DSS Audit Assistant API")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File processing utilities
def process_image(file_path: str) -> Tuple[str, str]:
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"
    with open(file_path, "rb") as f:
        return mime_type, base64.b64encode(f.read()).decode('utf-8')

def process_excel(file_path: str) -> str:
    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        return "\n".join([f"Sheet: {name}\n{xls.parse(name).to_string()}" for name in xls.sheet_names])
    except Exception as e:
        return f"Error processing Excel: {e}"

def excel_to_json(file_path: str) -> list:
    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        data = []
        for name in xls.sheet_names:
            df = xls.parse(name)
            for rec in df.to_dict(orient='records'):
                rec['sheet'] = name
                data.append(rec)
        return data
    except Exception as e:
        return [{"error": f"Error processing Excel: {e}"}]

def process_pdf(file_path: str) -> Tuple[str, List[str]]:
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        extracted_images = []
        for page in doc:
            for img in page.get_images(full=True):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.width > 300 and pix.height > 300:
                    img_name = f"temp_img_{xref}.png"
                    pix.save(img_name)
                    extracted_images.append(img_name)
                pix = None
        return full_text, extracted_images
    except Exception as e:
        return f"Error processing PDF: {e}", []

def process_files(paths: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    text, images = "", []
    for path in list(paths):
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(process_image(path))
        elif path.lower().endswith(('.xlsx', '.xls')):
            text += f"\n--- Excel File: {os.path.basename(path)} ---\n"
            text += process_excel(path) + "\n"
        elif path.lower().endswith('.pdf'):
            text += f"\n--- PDF File: {os.path.basename(path)} ---\n"
            pdf_text, pdf_imgs = process_pdf(path)
            text += pdf_text + "\n"
            for img in pdf_imgs:
                paths.append(img)
                images.append(process_image(img))
    return text, images

def extract_json_from_response(text: str) -> str:
    txt = text.strip()
    if txt.startswith("```json"):
        txt = txt[7:]
    if txt.startswith("```"):
        txt = txt[3:]
    if txt.endswith("```"):
        txt = txt[:-3]
    match = re.search(r'\[.*\]', txt, re.DOTALL)
    return match.group(0) if match else txt

def convert_datetime(obj):
    if isinstance(obj, dict):
        return {k: convert_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_datetime(i) for i in obj]
    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    return obj

@app.post("/analyze_asset_scope")
async def analyze_asset_scope(files: List[UploadFile] = File(...)):
    # Store files in temp dir
    temp_dir = tempfile.mkdtemp(prefix="pci_dss_")
    file_paths, excel_data = [], []
    for file in files:
        name = re.sub(r'[^\w\.-]', '_', os.path.basename(file.filename))
        path = os.path.join(temp_dir, name)
        with open(path, 'wb') as f:
            f.write(await file.read())
        file_paths.append(path)
        if name.lower().endswith(('.xlsx', '.xls')):
            excel_data = excel_to_json(path)

    # Process content
    context_text, context_images = process_files(file_paths)
    if not excel_data:
        raise HTTPException(status_code=400, detail="No Excel asset inventory found.")

    # Prepare prompts
    system_prompt = (
        "You are an expert PCI DSS auditor. Given the context from uploaded documents and the asset inventory as JSON, "
        "analyze each asset and add a new field 'scope' with value 'in-scope' or 'out-of-scope'. Return ONLY the modified inventory as JSON list."
    )
    detailed_prompt = (
        "You are an expert PCI DSS auditor. For each asset in the provided inventory, analyze all extracted context. "
        "Keep original fields, add 'Scope' (in-scope/out-of-scope) and 'Reason for scope'. Return only JSON list."
    )
    clean_data = convert_datetime(excel_data)
    content = [{
        "type": "text",
        "text": f"{system_prompt}\n\nContext:\n{context_text}\n\nAsset Inventory:\n{json.dumps(clean_data, indent=2)}\n\n{detailed_prompt}"
    }]
    for mime, b64 in context_images:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{"role": "user", "content": content}],
        temperature=0.3
    )
    raw = response.choices[0].message.content
    json_str = extract_json_from_response(raw)
    try:
        result = json.loads(json_str)
    except:
        result = raw
    return {"assets_with_scope": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
