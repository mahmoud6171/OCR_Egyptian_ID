# Egyptian ID Card OCR API

This project provides an API and web interface for extracting information from Egyptian ID cards using deep learning and OCR techniques.

## Features

- Upload images of Egyptian ID cards (front and back)
- Automatic detection and cropping of ID card regions
- Extraction of fields such as name, national ID, address, birth date, governorate, gender, serial, issue/expiry date, job, education, religion, and marital status
- Web frontend for easy interaction
- FastAPI backend with RESTful endpoints
- GPU acceleration support (if available)

## Project Structure

```
app_fastapi.py         # Main FastAPI backend
utils.py               # OCR, detection, and processing utilities
models/
static/
  index.html           # Web frontend
  main.js              # Frontend JavaScript
  ...

```

## Requirements

- Python 3.8+
- pip
- (Recommended) NVIDIA GPU with CUDA for best performance

### Python Packages

- fastapi
- uvicorn
- easyocr
- pytesseract
- opencv-python
- ultralytics
- torch (with CUDA if using GPU)
- numpy

## Installation & Setup

1. **Clone the repository:**

   ```sh
   git clone <repo-url>
   cd OCR_Egyptian_ID-main
   ```

2. **Create a virtual environment (optional but recommended):**

   ```sh
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # or
   source venv/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies:**

   ```sh
   pip install -r requirements.txt
   ```
   If you don't have a `requirements.txt`, install manually:
   ```sh
   pip install fastapi uvicorn easyocr pytesseract opencv-python ultralytics torch numpy
   ```
   For GPU support, follow [PyTorch installation instructions](https://pytorch.org/get-started/locally/).

4. **Install Tesseract OCR:**
   - Download and install from [Tesseract OCR releases](https://github.com/tesseract-ocr/tesseract).
   - Update the `pytesseract.pytesseract.tesseract_cmd` path in `utils.py` if needed.


## How to Run

1. **Start the FastAPI server:**

   ```sh
   uvicorn app_fastapi:app --reload
   ```
   The app will be available at [http://localhost:8000](http://localhost:8000)

2. **Open the web interface:**
   - Go to [http://localhost:8000](http://localhost:8000) in your browser.

## API Endpoints

- `POST /api/v1/ocr/front` — Process front side of ID card
- `POST /api/v1/ocr/back` — Process back side of ID card
- `GET /processed_image/{side}` — Get processed image (if available)
- `GET /health` — Health check

## Notes

- For best results, use high-quality images of ID cards.
- GPU acceleration is recommended for faster processing.
- The web interface is located in the `static/` folder.

## License

This project is for educational and research purposes.
