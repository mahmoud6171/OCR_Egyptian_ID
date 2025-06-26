from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
import tempfile
import os
from PIL import Image
import io
from utils import detect_and_process_id_card, detect_and_process_back_side
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
app = FastAPI(
    title="Egyptian ID Card OCR API",
    description="API for extracting information from Egyptian ID cards",
    version="1.0.0"
)

# Mount static files directory
static_files = StaticFiles(directory="static")
app.mount("/static", static_files, name="static")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create a thread pool for handling CPU-intensive OCR tasks
thread_pool = ThreadPoolExecutor(max_workers=4)

class OCRResponse(BaseModel):
    success: bool
    message: str
    data: dict = None
    processed_image_url: str = None

async def process_image_in_thread(process_func, temp_file_path):
    """Run CPU-intensive OCR processing in a thread pool"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(thread_pool, process_func, temp_file_path)

@app.post("/api/v1/ocr/front", response_model=OCRResponse)
async def process_front_side(file: UploadFile = File(...)):
    """Process front side of Egyptian ID card"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Process image in thread pool
        try:
            first_name, second_name, full_name, national_id, address, birth, gov, gender, serial = await process_image_in_thread(
                detect_and_process_id_card, temp_file_path
            )
            
            # Prepare response data
            data = {
                "first_name": first_name,
                "second_name": second_name,
                "full_name": full_name,
                "national_id": national_id,
                "address": address,
                "birth_date": birth,
                "governorate": gov,
                "gender": gender,
                "serial": serial
            }
            
            # Return processed image if available
            processed_image_path = "processed_image.jpg"
            if os.path.exists(processed_image_path):
                return OCRResponse(
                    success=True,
                    message="Front side processed successfully",
                    data=data,
                    processed_image_url="/processed_image/front"
                )
            
            return OCRResponse(
                success=True,
                message="Front side processed successfully",
                data=data
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        
        finally:
            # Cleanup
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling file: {str(e)}")

@app.post("/api/v1/ocr/back", response_model=OCRResponse)
async def process_back_side(file: UploadFile = File(...)):
    """Process back side of Egyptian ID card"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Process image in thread pool
        try:
            back_nid, issue_date, expiry_date, job, education, religion, marital_status = await process_image_in_thread(
                detect_and_process_back_side, temp_file_path
            )
            
            # Prepare response data
            data = {
                "back_nid": back_nid,
                "issue_date": issue_date,
                "expiry_date": expiry_date,
                "job": job,
                "education": education,
                "religion": religion,
                "marital_status": marital_status
            }
            
            # Return processed image if available
            processed_image_path = "d2.jpg"
            if os.path.exists(processed_image_path):
                return OCRResponse(
                    success=True,
                    message="Back side processed successfully",
                    data=data,
                    processed_image_url="/processed_image/back"
                )
            
            return OCRResponse(
                success=True,
                message="Back side processed successfully",
                data=data
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")
        
        finally:
            # Cleanup
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error handling file: {str(e)}")

@app.get("/processed_image/{side}")
async def get_processed_image(side: str):
    """Return the processed image"""
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend application"""
    with open("static/index.html") as f:
        return f.read()
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run("app_fastapi:app", host="localhost", port=8000, reload=True)
