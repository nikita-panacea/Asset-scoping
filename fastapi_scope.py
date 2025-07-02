from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import base64
from openai import OpenAI
import pandas as pd
import fitz
from PyPDF2 import PdfReader
from PIL import Image
import openpyxl
import uuid
import tempfile
from typing import Dict, List, Tuple
import time
import re
from dotenv import load_dotenv
import datetime
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
 
# Session management
sessions: Dict[str, dict] = {}
 
# Session timeout (1 hour)
#NOTE: Change as required
SESSION_TIMEOUT = 3600
 
# Session class to store data on FastAPI server
class SessionData:
    def __init__(self):
        self.processed_data: Tuple[str, List[Tuple[str, str]]] = ("", [])
        self.document_data: Dict[str, Tuple[str, str, List]] = {}
        self.chat_history: List[Tuple[str, str]] = []
        self.doc_chat_histories: Dict[str, List[Tuple[str, str]]] = {}
        self.reports: Dict[str, str] = {}
        self.last_access: float = time.time()
 
# Utility functions
def get_session(session_id: str) -> SessionData:
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    if time.time() - session.last_access > SESSION_TIMEOUT:
        del sessions[session_id]
        raise HTTPException(status_code=410, detail="Session expired")
    session.last_access = time.time()
    return session
 
# File processing functions (same as in Gradio app implementation)
 
# Process images to convert into base64 encoding
def process_image(file_path: str) -> Tuple[str, str]:
    ext = os.path.splitext(file_path)[1].lower()
    mime_type = "image/jpeg" if ext in ('.jpg', '.jpeg') else "image/png"
    with open(file_path, "rb") as f:
        return (mime_type, base64.b64encode(f.read()).decode('utf-8'))
 
# Process excel sheets to extract text
def process_excel(file_path: str) -> str:
    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        return "\n".join([f"Sheet: {name}\n{xls.parse(name).to_string()}" for name in xls.sheet_names])
    except Exception as e:
        return f"Error processing Excel: {e}"
 
# Convert excel sheets to JSON format
def excel_to_json(file_path: str) -> list:
    try:
        xls = pd.ExcelFile(file_path, engine='openpyxl')
        data = []
        for name in xls.sheet_names:
            df = xls.parse(name)
            records = df.to_dict(orient='records')
            for rec in records:
                rec['sheet'] = name
                data.append(rec)
        return data
    except Exception as e:
        return [{"error": f"Error processing Excel: {e}"}]
 
# Process PDF file to extract text and images
def process_pdf(file_path: str) -> Tuple[str, List[str]]:
    try:
        doc = fitz.open(file_path)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
 
        extracted_images = []
        for page in doc:
            image_list = page.get_images(full=True)
            for img in image_list:
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)
                if pix.width > 300 and pix.height > 300:
                    image_filename = f"temp_img_{xref}.png"
                    pix.save(image_filename)
                    extracted_images.append(image_filename)
                pix = None
        return full_text, extracted_images
    except Exception as e:
        return f"Error processing PDF: {e}", []
 
# Process uploaded files according to file type
def process_files(file_paths: List[str]) -> Tuple[str, List[Tuple[str, str]]]:
    text, images = "", []
    for path in file_paths:
        base = os.path.basename(path)
        if path.lower().endswith(('.png', '.jpg', '.jpeg')):
            images.append(process_image(path))
        elif path.lower().endswith(('.xlsx', '.xls')):
            text += f"\n--- Excel File: {base} ---\n"
            text += process_excel(path) + "\n"
        elif path.lower().endswith('.pdf'):
            text += f"\n--- PDF File: {base} ---\n"
            pdf_text, pdf_images = process_pdf(path)
            text += pdf_text + "\n"
            for img in pdf_images:
                file_paths.append(img)
                images.append(process_image(img))
    return text, images
 
# Function to send files (text and images) to GPT (to generate scope and chat)
def analyze_with_gpt4o(prompt: str, text: str, images: List[Tuple[str, str]], history=None) -> str:
    content = [{"type": "text", "text": f"Context:\n{text}\n\nQuestion: {prompt}"}]
    for mime, b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
   
    messages = history.copy() if history else []
    messages.append({"role": "user", "content": content})
   
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content
 
# Utility function to convert datetime objects to ISO format
def convert_datetime(obj):
    if isinstance(obj, dict):
        return {k: convert_datetime(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_datetime(i) for i in obj]
    elif isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    else:
        return obj
 
# API Endpoints
# creates a new session with unique id
# the session data is stored in memory within the FastAPI application's runtime
# The sessions dictionary stores all session data in memory.
# Each session is identified by a unique session_id (UUID).
@app.post("/create_session")
async def create_session():
    session_id = str(uuid.uuid4())
    sessions[session_id] = SessionData()
    return {"session_id": session_id}
 
# Endpoint to upload files(Image, PDF, Excel) and process them (extract text and images)
# creates pointers to file paths, extract images with file
@app.post("/upload_files")
async def upload_files(session_id: str, files: List[UploadFile] = File(...)):
    try:
        session = get_session(session_id)
    except HTTPException as e:
        return e
 
    # Create session-specific storage directory
    if not hasattr(session, 'session_dir'):
        session.session_dir = os.path.join(tempfile.gettempdir(), f"pci_dss_{session_id}")
        os.makedirs(session.session_dir, exist_ok=True)
 
    processed_files = []
    for file in files:
        try:
            # get original filename
            original_name = os.path.basename(file.filename)
            clean_name = re.sub(r'[^\w\.-]', '_', original_name)
           
            # Handle duplicate names
            base, ext = os.path.splitext(clean_name)
            counter = 0
            final_name = clean_name
            while final_name in session.document_data:
                counter += 1
                final_name = f"{base}_{counter}{ext}"
           
            # Save file with final name
            file_path = os.path.join(session.session_dir, final_name)
            with open(file_path, 'wb') as f:
                content = await file.read()
                f.write(content)
           
            # Process file content according to file type
            text, images = "", []
            if final_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                images.append(process_image(file_path))
                # Update document_data for the Image
                session.document_data[final_name] = {
                    "original_name": original_name,
                    "path": file_path,
                    "text": '',
                    "images": images,
                    "type": "IMAGE"
                }
            elif final_name.lower().endswith(('.xlsx', '.xls')):
                text = process_excel(file_path)
                excel_json = excel_to_json(file_path)  
                # Update document_data for the Excel
                session.document_data[final_name] = {
                    "original_name": original_name,
                    "path": file_path,
                    "text": text,
                    "images": [],
                    "type": "EXCELSHEET",
                    "excel_json": excel_json 
                }
            # PDF-specific processing    
            elif final_name.lower().endswith('.pdf'):
                text, pdf_images = process_pdf(file_path)
                # Process extracted images
                for i, img_path in enumerate(pdf_images):
                    try:
                        # Generate image name based on PDF
                        pdf_base = os.path.splitext(final_name)[0]
                        img_original = f"Image {i+1} from {original_name}"
                        img_clean = f"{pdf_base}_image_{i+1}.png"
                       
                        # Handle duplicates
                        counter = 0
                        while img_clean in session.document_data:
                            counter += 1
                            img_clean = f"{pdf_base}_image_{i+1}_{counter}.png"
                       
                        # Move image to session directory
                        new_img_path = os.path.join(session.session_dir, img_clean)
                        os.rename(img_path, new_img_path)
                       
                        # Process and store image metadata
                        images = [process_image(new_img_path)]
                        session.document_data[img_clean] = {
                            "original_name": img_original,
                            "path": new_img_path,
                            "text": "",
                            "images": images,
                            "parent_pdf": final_name,
                            "type": "IMAGE"
                        }
                        processed_files.append(img_clean)
                    except Exception as e:
                        print(f"Error processing extracted image: {str(e)}")
 
                # Update document_data for the PDF
                session.document_data[final_name] = {
                    "original_name": original_name,
                    "path": file_path,
                    "text": text,
                    "images": [],
                    "type": "PDF"
                }
            processed_files.append(final_name)
 
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error processing {file.filename}: {str(e)}"
            )
 
    # Process combined files data
    all_paths = [v['path'] for v in session.document_data.values()]
    combined_text, combined_images = process_files(all_paths)
    session.processed_data = (combined_text, combined_images)
 
    return {
        "message": "Files processed successfully",
        "documents": [{
            "id": name,
            "original_name": data['original_name'],
            "type": data.get('type', 'FILE'),
            "parent": data.get('parent_pdf')
        } for name, data in session.document_data.items()]
    }
 
import json
 
def extract_json_from_response(text):
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    match = re.search(r'\[.*\]', text, re.DOTALL)
    if match:
        return match.group(0)
    return text
 
@app.post("/analyze_asset_scope")
async def analyze_asset_scope_api(session_id: str):
    try:
        session = get_session(session_id)
    except HTTPException as e:
        return e
 
    # Find the first Excel file's JSON (or let user select)
    excel_json = []
    for doc in session.document_data.values():
        if doc.get("type") == "EXCELSHEET" and "excel_json" in doc:
            excel_json = doc["excel_json"]
            break
    if not excel_json:
        return JSONResponse(status_code=400, content={"error": "No Excel asset inventory found."})
 
    context_text, images = session.processed_data
    prompt="""
    You are an expert PCI DSS auditor.For each asset in the provided asset inventory (see JSON list below), analyze all the extracted context from the uploaded documents (including network diagrams, data flows, and any other relevant information).

    For each asset:
    - Keep all original fields and their values exactly as they appear in the input.
    - Add a new field called "Scope" to each asset object, with the value "in-scope" or "out-of-scope" based on your analysis.
    -Also add more field "Reason for scope", denoting why it is in-scope or out-of-scope.

    Return ONLY the modified asset inventory as a JSON list, with all original fields and values for each asset, plus the new "scope" field for each asset. Do not summarize, group, or omit any assets or fields. The output must be a JSON array with all assets, each as a full object.Don't need to return the metadata that is not containing the PCI DSS relevant asset.
    """
 
    # Compose system prompt and call OpenAI
    system_prompt = (
        "You are an expert PCI DSS auditor. "
        "Given the following context extracted from uploaded documents (network diagrams, data flows, etc.) "
        "and the asset inventory (as JSON), analyze each asset and add a new field 'scope' with value 'in-scope' or 'out-of-scope'. "
        "Return ONLY the modified asset inventory as a JSON list."
    )
    # Convert datetimes before dumping to JSON
    excel_json_clean = convert_datetime(excel_json)
    content = [
        {"type": "text", "text": f"{system_prompt}\n\nContext:\n{context_text}\n\nAsset Inventory:\n{json.dumps(excel_json_clean, indent=2)}\n\n{prompt}"}
    ]
    for mime, b64 in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
    messages = [{"role": "user", "content": content}]
    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        temperature=0.3
    )
    raw_content = response.choices[0].message.content
    json_str = extract_json_from_response(raw_content)
    try:
        result_json = json.loads(json_str)
    except Exception:
        result_json = raw_content
 
    return {"assets_with_scope": result_json}
 
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app)