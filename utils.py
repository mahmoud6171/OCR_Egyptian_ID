from ultralytics import YOLO
import cv2
import re
import easyocr
import pytesseract
import numpy as np
import os
from functools import lru_cache

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Users\Mahmoud.Ryad\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

# Cache for YOLO models
@lru_cache(maxsize=None)
def get_yolo_model(model_name):
    model = YOLO(model_name)
    if gpu_available:
        model.to('cuda')  # Move model to GPU
    return model

# Initialize GPU settings
def check_gpu_status():
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA is not available. Make sure you have an NVIDIA GPU and CUDA installed.")
            print("Current PyTorch CUDA status:", torch.version.cuda)
            return False
        
        # Test CUDA capabilities
        try:
            torch.cuda.current_device()
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            
            print(f"✅ GPU Successfully Detected:")
            print(f"   - Number of GPUs: {device_count}")
            print(f"   - GPU Model: {device_name}")
            print(f"   - CUDA Version: {cuda_version}")
            
            # Enable cuDNN auto-tuner
            torch.backends.cudnn.benchmark = True
            return True
            
        except Exception as e:
            print(f"⚠️ Error initializing CUDA: {str(e)}")
            return False
            
    except ImportError:
        print("⚠️ PyTorch is not installed properly. Installing GPU version...")
        print("Please run: pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118")
        return False

gpu_available = check_gpu_status()

# Initialize EasyOCR readers with caching
@lru_cache(maxsize=None)
def get_reader(lang='ar'):
    # Always include both Arabic and English for better recognition
    languages = ['ar', 'en']
    if lang == 'en':
        # If English is specifically requested, use English first
        return easyocr.Reader(['en'], gpu=gpu_available, )
    
    return easyocr.Reader(languages, gpu=gpu_available, detect_network="craft")

# Global model instances
id_card_model = get_yolo_model('models/detect_id_card.pt')
id_number_model = get_yolo_model('models/detect_id.pt')
fields_model = get_yolo_model('models/detect_odjects.pt')

# Function to preprocess the cropped image
def preprocess_image(cropped_image):
    gray_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)   
    return gray_image

# Functions for specific fields with custom OCR configurations
def extract_text(image, bbox, lang='ar'):
    x1, y1, x2, y2 = bbox
    cropped_image = image[y1:y2, x1:x2]
    preprocessed_image = preprocess_image(cropped_image)
    reader = get_reader(lang)
    results = reader.readtext(preprocessed_image, detail=0, paragraph=True)
    text = ' '.join(results) if results else ''
    return text.strip()

# Function to detect national ID numbers in a cropped image
def detect_national_id(cropped_image):
    results = id_number_model(cropped_image)
    detected_info = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_info.append((cls, x1))
            cv2.rectangle(cropped_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cropped_image, str(cls), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

    detected_info.sort(key=lambda x: x[1])
    id_number = ''.join([str(cls) for cls, _ in detected_info])
    
    return id_number

# Function to remove numbers from a string
def remove_numbers(text):
    return re.sub(r'\d+', '', text)

# Function to expand bounding box height only
def expand_bbox_height(bbox, scale=1.2, image_shape=None):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    center_x = x1 + width // 2
    center_y = y1 + height // 2
    new_height = int(height * scale)
    new_y1 = max(center_y - new_height // 2, 0)
    new_y2 = min(center_y + new_height // 2, image_shape[0])
    return [x1, new_y1, x2, new_y2]

# Function to process the cropped image
def process_image(cropped_image):
    results = fields_model(cropped_image)

    # Variables to store extracted values
    first_name = ''
    second_name = ''
    merged_name = ''
    nid = ''
    address = ''
    serial = ''

    # Loop through the results
    for result in results:
        output_path = 'processed_image.jpg'
        result.save(output_path)

        for box in result.boxes:
            bbox = box.xyxy[0].tolist()
            class_id = int(box.cls[0].item())
            class_name = result.names[class_id]
            bbox = [int(coord) for coord in bbox]

            if class_name == 'firstName':
                first_name = extract_text(cropped_image, bbox, lang='ar')
            elif class_name == 'lastName':
                second_name = extract_text(cropped_image, bbox, lang='ar')
            elif class_name == 'serial':
                serial = extract_text(cropped_image, bbox, lang='en')
            elif class_name == 'address':
                address = extract_text(cropped_image, bbox, lang='ar')
            elif class_name == 'nid':
                expanded_bbox = expand_bbox_height(bbox, scale=1.5, image_shape=cropped_image.shape)
                cropped_nid = cropped_image[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
                nid = detect_national_id(cropped_nid)

    merged_name = f"{first_name} {second_name}"
    print(f"First Name: {first_name}")
    print(f"Second Name: {second_name}")
    print(f"Full Name: {merged_name}")
    print(f"National ID: {nid}")
    print(f"Address: {address}")
    print(f"Serial: {serial}")

    decoded_info = decode_egyptian_id(nid)
    return (first_name, second_name, merged_name, nid, address, decoded_info["Birth Date"], decoded_info["Governorate"], decoded_info["Gender"],serial)

# Function to decode the Egyptian ID number
def decode_egyptian_id(id_number):
    governorates = {
        '01': 'Cairo',
        '02': 'Alexandria',
        '03': 'Port Said',
        '04': 'Suez',
        '11': 'Damietta',
        '12': 'Dakahlia',
        '13': 'Ash Sharqia',
        '14': 'Kaliobeya',
        '15': 'Kafr El - Sheikh',
        '16': 'Gharbia',
        '17': 'Monoufia',
        '18': 'El Beheira',
        '19': 'Ismailia',
        '21': 'Giza',
        '22': 'Beni Suef',
        '23': 'Fayoum',
        '24': 'El Menia',
        '25': 'Assiut',
        '26': 'Sohag',
        '27': 'Qena',
        '28': 'Aswan',
        '29': 'Luxor',
        '31': 'Red Sea',
        '32': 'New Valley',
        '33': 'Matrouh',
        '34': 'North Sinai',
        '35': 'South Sinai',
        '88': 'Foreign'
    }

    century_digit = int(id_number[0])
    year = int(id_number[1:3])
    month = int(id_number[3:5])
    day = int(id_number[5:7])
    governorate_code = id_number[7:9]
    gender_code = int(id_number[12:13])

    if century_digit == 2:
        century = "1900-1999"
        full_year = 1900 + year
    elif century_digit == 3:
        century = "2000-2099"
        full_year = 2000 + year
    else:
        raise ValueError("Invalid century digit")

    gender = "Male" if gender_code % 2 != 0 else "Female"
    governorate = governorates.get(governorate_code, "Unknown")
    birth_date = f"{full_year:04d}-{month:02d}-{day:02d}"

    return {
        'Birth Date': birth_date,
        'Governorate': governorate,
        'Gender': gender
    }

# Function to detect the ID card and pass it to the existing code
def detect_and_process_id_card(image_path):
    # Perform inference to detect the ID card
    id_card_results = id_card_model(image_path)

    # Load the original image using OpenCV
    image = cv2.imread(image_path)

    # Crop the ID card from the image
    for result in id_card_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]

    # Pass the cropped image to the existing processing function
    return process_image(cropped_image)

# print(detect_and_process_id_card("font_ID.jpg"))

def convert_arabic_to_english_numbers(text):
    """Convert Arabic/Persian numbers to English numbers"""
    arabic_to_english = {
        '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
        '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9'
    }
    return ''.join(arabic_to_english.get(char, char) for char in text)

def clean_and_extract_numbers(text):
    """Extract numbers from text, handling both Arabic and English numerals"""
    # First convert Arabic numbers to English
    text = convert_arabic_to_english_numbers(text)
    # Then extract only the digits
    return ''.join(char for char in text if char.isdigit())

# Function to detect numbers in a cropped image
def detect_numbers(cropped_image):
    model = id_number_model
    results = model(cropped_image)
    detected_info = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detected_info.append((cls, x1))

    detected_info.sort(key=lambda x: x[1])
    numbers = ''.join([str(cls) for cls, _ in detected_info])
    
    return numbers

def is_valid_year(year):
    """Check if the year is valid (between 1900 and 2100)"""
    try:
        year_num = int(year)
        return 1900 <= year_num <= 2100
    except ValueError:
        return False

def is_valid_month(month):
    """Check if the month is valid (1-12)"""
    try:
        month_num = int(month)
        return 1 <= month_num <= 12
    except ValueError:
        return False

def is_valid_day(day):
    """Check if the day is valid (1-31)"""
    try:
        day_num = int(day)
        return 1 <= day_num <= 31
    except ValueError:
        return False

def extract_date_with_tesseract(image):
    """
    Extract date using Tesseract OCR with specific configuration for Arabic numbers
    """
    # Preprocess the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to get black text on white background
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Apply dilation to make text thicker and clearer
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    
    # Save debug image
    cv2.imwrite('tesseract_input.jpg', dilated)
    
    # Configure Tesseract for Arabic and numbers only
    custom_config = r'--oem 3 --psm 6 -l ara+eng -c tessedit_char_whitelist=0123456789٠١٢٣٤٥٦٧٨٩/:'
    
    try:
        # Get OCR text
        text = pytesseract.image_to_string(dilated, config=custom_config)
        print(f"Tesseract raw output: {text}")
        
        # Clean and process the text
        text = text.strip()
        if text:
            return extract_date_from_ocr_text(text)
    except Exception as e:
        print(f"Tesseract error: {e}")
    
    return None

def extract_date_from_ocr_text(text):
    """
    Extract date from OCR text that might contain numbers in various formats
    """
    print(f"Trying to extract date from OCR text: {text}")
    
    # Convert to English numbers for processing
    text = convert_arabic_to_english_numbers(text)
    print(f"Converted to English numbers: {text}")
    
    # Clean the text - remove common separators and extra spaces
    text = re.sub(r'[:/\-_،]+', ' ', text)
    parts = [part.strip() for part in text.split() if part.strip()]
    print(f"Split parts: {parts}")
    
    # Try different date patterns
    for i, part in enumerate(parts):
        # Pattern 1: Look for 4-digit year
        if len(part) == 4 and part.isdigit():
            year = int(part)
            if 2000 <= year <= 2100:
                # Look for month in adjacent parts
                for j in range(max(0, i-1), min(len(parts), i+2)):
                    if i != j and parts[j].isdigit():
                        month = int(parts[j])
                        if 1 <= month <= 12:
                            return f"{year}-{month:02d}-01"
        
        # Pattern 2: Look for 2-digit year with 20 prefix
        elif len(part) == 2 and part.isdigit():
            year = 2000 + int(part)
            if 2000 <= year <= 2100:
                # Look for month in adjacent parts
                for j in range(max(0, i-1), min(len(parts), i+2)):
                    if i != j and parts[j].isdigit():
                        month = int(parts[j])
                        if 1 <= month <= 12:
                            return f"{year}-{month:02d}-01"
        
        # Pattern 3: Combined year and month (YYYYMM)
        elif len(part) == 6 and part.isdigit():
            year = int(part[:4])
            month = int(part[4:])
            if 2000 <= year <= 2100 and 1 <= month <= 12:
                return f"{year}-{month:02d}-01"
        
        # Pattern 4: Try reversing numbers for RTL text
        reversed_part = part[::-1]
        if len(reversed_part) == 4 and reversed_part.isdigit():
            year = int(reversed_part)
            if 2000 <= year <= 2100:
                # Look for month in adjacent parts (also try reversed)
                for j in range(max(0, i-1), min(len(parts), i+2)):
                    if i != j:
                        adj_part = parts[j]
                        # Try both original and reversed
                        for test_part in [adj_part, adj_part[::-1]]:
                            if test_part.isdigit():
                                month = int(test_part)
                                if 1 <= month <= 12:
                                    return f"{year}-{month:02d}-01"
    
    return None

def detect_date_numbers(cropped_image):
    """
    Use the trained number detection model to detect individual numbers in the date field.
    Returns the detected numbers sorted by x-coordinate.
    """
    # Ensure minimum dimensions and proper scaling
    min_height = 160  # Model seems to work better with images at least this tall
    
    # Calculate scaling factor if needed
    height, width = cropped_image.shape[:2]
    if height < min_height:
        scale = min_height / height
        new_width = int(width * scale)
        cropped_image = cv2.resize(cropped_image, (new_width, min_height))
        print(f"Resized image from {height}x{width} to {min_height}x{new_width}")
    
    # Add padding if the image is too narrow
    min_width = min_height * 2  # Ensure reasonable aspect ratio
    if cropped_image.shape[1] < min_width:
        right_padding = min_width - cropped_image.shape[1]
        cropped_image = cv2.copyMakeBorder(
            cropped_image, 
            0, 0, 
            0, right_padding, 
            cv2.BORDER_CONSTANT, 
            value=(255, 255, 255)
        )
        print(f"Added {right_padding}px right padding to maintain aspect ratio")

    # Save debug image
    cv2.imwrite('debug_date_region.jpg', cropped_image)
    print(f"Saved debug image with shape {cropped_image.shape}")

    model = id_number_model
    results = model(cropped_image)
    detected_info = []

    for result in results:
        for box in result.boxes:
            cls = int(box.cls)
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf)
            detected_info.append((cls, x1, conf))
            print(f"Detected number {cls} at x={x1} with confidence {conf:.2f}")

    # Sort by x-coordinate to get numbers in order
    detected_info.sort(key=lambda x: x[1])
    numbers = [str(cls) for cls, _, _ in detected_info]
    print(f"Detected numbers in order: {numbers}")
    return numbers

def extract_date_with_easyocr(image):
    """
    Extract date using EasyOCR
    """
    try:
        # Initialize reader with Arabic and English
        reader = get_reader('ar')
        
        # Get OCR text
        results = reader.readtext(image, detail=0)
        text = ' '.join(results)
        print(f"EasyOCR raw output: {text}")
        return text.strip()
    except Exception as e:
        print(f"EasyOCR error: {e}")
        return None

def binarize_image(image):
    """
    Apply advanced binarization to make text clearer
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
        
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,  # Block size
        2    # C constant
    )
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((2,2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def clean_and_extract_numbers_with_order(text):
    """Extract numbers from text while preserving their order"""
    # Convert Arabic numbers to English
    text = convert_arabic_to_english_numbers(text)
    # Split by spaces to preserve order
    parts = text.split()
    # Keep only the numeric parts in their original order
    numbers = [part for part in parts if part.isdigit()]
    return numbers

def format_date_numbers(numbers):
    """
    Format the detected numbers into a proper date string.
    Tries both MM/YYYY and YYYY/MM formats.
    """
    if not numbers:
        return None, None

    # Convert all numbers to strings and join them
    numbers = [str(n) for n in numbers]
    
    # Try to identify month and year parts
    potential_dates = []
    
    # First, try to find a 4-digit year
    for i, num in enumerate(numbers):
        if len(num) == 4 and num.startswith('20'):  # Found a potential year
            year = num
            # Look for a 1 or 2 digit month before or after the year
            if i > 0:  # Check numbers before the year
                month = numbers[i-1]
                if len(month) <= 2:  # Month should be 1 or 2 digits
                    try:
                        month_num = int(month)
                        if 1 <= month_num <= 12:
                            potential_dates.append((f"{month:02d}/{year}", f"{year}/{month:02d}"))
                    except ValueError:
                        continue
            if i < len(numbers) - 1:  # Check numbers after the year
                month = numbers[i+1]
                if len(month) <= 2:  # Month should be 1 or 2 digits
                    try:
                        month_num = int(month)
                        if 1 <= month_num <= 12:
                            potential_dates.append((f"{month:02d}/{year}", f"{year}/{month:02d}"))
                    except ValueError:
                        continue
    
    # If no 4-digit year found, try combining numbers
    if not potential_dates:
        # Try combining adjacent numbers
        for i in range(len(numbers) - 1):
            combined = ''.join(numbers[i:i+2])
            if len(combined) == 4 and combined.startswith('20'):  # Found a potential year
                year = combined
                # Look for month in remaining numbers
                for j, num in enumerate(numbers):
                    if j < i or j >= i+2:  # Only look at numbers not used in year
                        if len(num) <= 2:  # Month should be 1 or 2 digits
                            try:
                                month_num = int(num)
                                if 1 <= month_num <= 12:
                                    potential_dates.append((f"{month_num:02d}/{year}", f"{year}/{month_num:02d}"))
                            except ValueError:
                                continue

    if potential_dates:
        # Return both formats (MM/YYYY and YYYY/MM)
        return potential_dates[0]
    
    return None, None

def enhance_image_for_ocr(image):
    """
    Apply image enhancement techniques to improve OCR accuracy.
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Increase contrast using CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced)
    
    # Sharpen the image
    kernel = np.array([[-1,-1,-1],
                      [-1, 9,-1],
                      [-1,-1,-1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)
    
    return sharpened

def process_date_region(date_region):
    """
    Process a date region by detecting individual number components and performing OCR on each.
    Returns a list of detected numbers in left-to-right order.
    """
    debug_dir = 'debug_date_numbers'
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
    
    print("\n=== Processing Date Region ===")
    print(f"Input region shape: {date_region.shape}")
    
    # Save original region
    cv2.imwrite(os.path.join(debug_dir, '1_original.jpg'), date_region)
    
    # Enhance the image
    enhanced = enhance_image_for_ocr(date_region)
    cv2.imwrite(os.path.join(debug_dir, '2_enhanced.jpg'), enhanced)
    
    # Try direct OCR on the enhanced region first
    try:
        print("\nTrying direct OCR on enhanced region:")
        # Try Tesseract first
        tesseract_text = pytesseract.image_to_string(
            enhanced,
            config='--oem 0 -l ara+eng --psm 6'
        ).strip()
        print(f"Tesseract full region: {tesseract_text}")
        
        # Try EasyOCR
        easyocr_reader = get_reader('ar')
        easyocr_results = easyocr_reader.readtext(enhanced, detail=0)
        print(f"EasyOCR full region: {easyocr_results}")
        
        # Extract numbers from full region results
        full_region_numbers = []
        if tesseract_text:
            tesseract_numbers = clean_and_extract_numbers(tesseract_text)
            if tesseract_numbers:
                full_region_numbers.append(tesseract_numbers)
        
        if easyocr_results:
            for result in easyocr_results:
                numbers = clean_and_extract_numbers(result)
                if numbers:
                    full_region_numbers.append(numbers)
        
        if full_region_numbers:
            print(f"Numbers from full region: {full_region_numbers}")
            # If we found valid numbers, return them
            for numbers in full_region_numbers:
                if len(numbers) >= 4:  # At least a year
                    return [numbers]
    except Exception as e:
        print(f"Error in full region OCR: {e}")
    
    # If full region OCR didn't work, try individual number detection
    print("\nTrying individual number detection:")
    
    # Try different thresholding methods
    # 1. Otsu's thresholding
    _, otsu_binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite(os.path.join(debug_dir, '3a_otsu_binary.jpg'), otsu_binary)
    
    # 2. Adaptive thresholding
    adaptive_binary = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        15,
        8
    )
    cv2.imwrite(os.path.join(debug_dir, '3b_adaptive_binary.jpg'), adaptive_binary)
    
    # 3. Local thresholding using Sauvola's method
    window_size = 25
    k = 0.2
    R = 128
    
    # Calculate mean and standard deviation
    mean = cv2.boxFilter(enhanced.astype(float), -1, (window_size, window_size), borderType=cv2.BORDER_REPLICATE)
    mean_square = cv2.boxFilter(enhanced.astype(float)**2, -1, (window_size, window_size), borderType=cv2.BORDER_REPLICATE)
    std = np.sqrt(mean_square - mean**2)
    
    # Calculate threshold
    threshold = mean * (1 + k * ((std / R) - 1))
    sauvola_binary = np.zeros_like(enhanced)
    sauvola_binary[enhanced > threshold] = 255
    cv2.imwrite(os.path.join(debug_dir, '3c_sauvola_binary.jpg'), sauvola_binary)
    
    # Try all binary images
    binaries = [
        ('otsu', otsu_binary),
        ('adaptive', adaptive_binary),
        ('sauvola', sauvola_binary)
    ]
    
    best_numbers = []
    
    for binary_name, binary in binaries:
        print(f"\nProcessing {binary_name} binary:")
        
        # Clean up binary image
        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        print(f"Found {num_labels-1} connected components")
        
        # Create visualization image
        vis_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
        
        # Store regions with x-coordinates
        number_regions = []
        
        # Process each component
        for i in range(1, num_labels):
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter by size and aspect ratio
            if (30 <= area <= 2000 and 
                w >= 5 and 
                h >= 10 and
                0.2 <= w/h <= 5.0):
                
                # Add padding
                pad = 5
                y1 = max(0, y - pad)
                y2 = min(binary.shape[0], y + h + pad)
                x1 = max(0, x - pad)
                x2 = min(binary.shape[1], x + w + pad)
                
                # Extract region from enhanced image
                region = enhanced[y1:y2, x1:x2]
                number_regions.append((x1, region))
                
                # Draw bounding box
                cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(vis_img, str(len(number_regions)), (x1, y1-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Save visualization
        cv2.imwrite(os.path.join(debug_dir, f'4_{binary_name}_regions.jpg'), vis_img)
        
        if not number_regions:
            print("No regions found")
            continue
        
        # Sort by x-coordinate
        number_regions.sort(key=lambda x: x[0])
        print(f"Found {len(number_regions)} potential number regions")
        
        # Initialize OCR readers
        easyocr_reader = get_reader('ar')
        detected_numbers = []
        
        # Process each region
        for i, (_, region) in enumerate(number_regions):
            # Save individual number image
            region_path = os.path.join(debug_dir, f'5_{binary_name}_number_{i+1}.jpg')
            cv2.imwrite(region_path, region)
            
            try:
                # Try EasyOCR first
                easyocr_results = easyocr_reader.readtext(region, detail=0)
                
                # Try Tesseract with different PSM modes
                tesseract_configs = [
                    '--oem 0 -l ara+eng --psm 10',  # Treat as single character
                    '--oem 0 -l ara+eng --psm 6',   # Assume uniform block of text
                    '--oem 0 -l ara+eng --psm 7'    # Treat as single line
                ]
                
                print(f"\nRegion {i+1}:")
                
                if easyocr_results:
                    print(f"  EasyOCR output: {easyocr_results[0]}")
                    number = clean_and_extract_numbers(easyocr_results[0])
                    if number:
                        detected_numbers.append(number)
                        print(f"  EasyOCR extracted number: {number}")
                
                for config in tesseract_configs:
                    tesseract_text = pytesseract.image_to_string(
                        region,
                        config=config
                    ).strip()
                    
                    if tesseract_text:
                        print(f"  Tesseract output ({config}): {tesseract_text}")
                        tesseract_number = clean_and_extract_numbers(tesseract_text)
                        if tesseract_number and tesseract_number not in detected_numbers:
                            detected_numbers.append(tesseract_number)
                            print(f"  Tesseract extracted number: {tesseract_number}")
                            break  # Stop if we found a number
                            
            except Exception as e:
                print(f"Error processing region {i+1}: {e}")
                continue
        
        if detected_numbers:
            best_numbers = detected_numbers
            print(f"\nFound numbers using {binary_name} binary: {detected_numbers}")
            break
    
    return best_numbers

def process_back_image(cropped_image):
    """
    Process the back side image of ID card.
    """
    print("\n=== Starting Image Processing ===")
    
    results = fields_model(cropped_image)

    # Variables to store extracted values
    back_nid = None
    issue_date = None
    expiry_date = None
    job = None
    education = None
    religion = None
    marital_status = None
    date_numbers = []
    
    # Initialize OCR reader
    reader = get_reader('ar')

    if results and len(results) > 0:
        print("\n=== Processing YOLO Detections ===")
        output_path = 'd2.jpg'
        results[0].save(output_path)

        # Process each detected field
        for result in results:
            boxes = result.boxes
            for box in boxes:
                bbox = box.xyxy[0].tolist()
                class_id = int(box.cls[0].item())
                class_name = result.names[class_id]
                bbox = [int(coord) for coord in bbox]
                
                if class_name == 'issue':
                    print(f"\nProcessing issue date field")
                    
                    # Extract date region with padding
                    x1, y1, x2, y2 = bbox
                    padding = int((y2 - y1) * 0.2)
                    y1 = max(0, y1 - padding)
                    y2 = min(cropped_image.shape[0], y2 + padding)
                    x1 = max(0, x1 - padding)
                    x2 = min(cropped_image.shape[1], x2 + padding)
                    date_region = cropped_image[y1:y2, x1:x2]
                    
                    # Create debug directory
                    debug_dir = 'debug_date_numbers'
                    if not os.path.exists(debug_dir):
                        os.makedirs(debug_dir)
                    
                    # Save original date region
                    cv2.imwrite(os.path.join(debug_dir, '1_date_region.jpg'), date_region)
                    
                    # Convert to grayscale
                    gray = cv2.cvtColor(date_region, cv2.COLOR_BGR2GRAY)
                    
                    # Apply thresholding
                    binary = cv2.adaptiveThreshold(
                        gray,
                        255,
                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY_INV,
                        15,
                        8
                    )
                    cv2.imwrite(os.path.join(debug_dir, '2_binary.jpg'), binary)
                    
                    # Find connected components
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    
                    # Create debug image
                    debug_img = cv2.cvtColor(binary.copy(), cv2.COLOR_GRAY2BGR)
                    
                    # Store regions with x-coordinates
                    number_regions = []
                    
                    # Process each component
                    for i in range(1, num_labels):
                        x = stats[i, cv2.CC_STAT_LEFT]
                        y = stats[i, cv2.CC_STAT_TOP]
                        w = stats[i, cv2.CC_STAT_WIDTH]
                        h = stats[i, cv2.CC_STAT_HEIGHT]
                        area = stats[i, cv2.CC_STAT_AREA]
                        
                        # Filter components by size and aspect ratio
                        if (30 <= area <= 2000 and 
                            w >= 5 and 
                            h >= 10 and
                            0.2 <= w/h <= 5.0):
                            
                            # Extract region with padding
                            pad = 5
                            y1 = max(0, y - pad)
                            y2 = min(binary.shape[0], y + h + pad)
                            x1 = max(0, x - pad)
                            x2 = min(binary.shape[1], x + w + pad)
                            
                            region = date_region[y1:y2, x1:x2]
                            number_regions.append((x1, region))
                            
                            # Draw bounding box
                            cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
                            cv2.putText(debug_img, str(len(number_regions)), (x1, y1-5),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    
                    # Save debug image with regions
                    cv2.imwrite(os.path.join(debug_dir, '3_regions.jpg'), debug_img)
                    
                    # Sort regions by x-coordinate
                    number_regions.sort(key=lambda x: x[0])
                    
                    # Process each region
                    for i, (x, region) in enumerate(number_regions):
                        # Save individual number image
                        cv2.imwrite(os.path.join(debug_dir, f'4_number_{i+1}.jpg'), region)
                        
                        try:
                            # Perform raw OCR on the region
                            results = reader.readtext(region, detail=0)
                            
                            if results:
                                raw_text = results[0]
                                print(f"\nRegion {i+1}:")
                                print(f"  Raw OCR output: {raw_text}")
                                
                                # Extract numbers
                                number = clean_and_extract_numbers(raw_text)
                                if number:
                                    date_numbers.append(number)
                                    print(f"  Extracted number: {number}")
                        except Exception as e:
                            print(f"Error processing region {i+1}: {e}")
                            continue
                    
                    print("\n=== Date Numbers (Left to Right) ===")
                    print(f"Numbers: {date_numbers}")
                    
                    # Format the date
                    date_mm_yyyy, date_yyyy_mm = format_date_numbers(date_numbers)
                    if date_mm_yyyy:
                        print(f"\nFormatted dates:")
                        print(f"MM/YYYY: {date_mm_yyyy}")
                        print(f"YYYY/MM: {date_yyyy_mm}")
                        
                        # Use YYYY/MM format for issue_date
                        issue_date = date_yyyy_mm.replace('/', '-')
                        
                        # Calculate expiry date
                        try:
                            from datetime import datetime, timedelta
                            year, month = map(int, issue_date.split('-'))
                            issue_dt = datetime(year, month, 1)
                            expiry_dt = issue_dt + timedelta(days=7*365)
                            expiry_date = expiry_dt.strftime("%Y-%m-%d")
                        except (ValueError, IndexError) as e:
                            print(f"Error calculating expiry date: {e}")
                            expiry_date = None
                
                elif class_name == 'nid_back':
                    expanded_bbox = expand_bbox_height(bbox, scale=1.5, image_shape=cropped_image.shape)
                    cropped_nid = cropped_image[expanded_bbox[1]:expanded_bbox[3], expanded_bbox[0]:expanded_bbox[2]]
                    detected_id = detect_national_id(cropped_nid)
                    if detected_id:
                        back_nid = detected_id
                    else:
                        id_text = extract_text(cropped_image, bbox, lang='ara')
                        if id_text:
                            numbers = clean_and_extract_numbers(id_text)
                            if len(numbers) >= 14:
                                back_nid = numbers[:14]
                
                elif class_name == 'demo':
                    demo_text = extract_text(cropped_image, bbox, lang='ara')
                    if demo_text:
                        if 'مسلم' in demo_text:
                            religion = 'مسلمة' if 'مسلمة' in demo_text else 'مسلم'
                        elif 'مسيحي' in demo_text:
                            religion = 'مسيحية' if 'مسيحية' in demo_text else 'مسيحي'
                        
                        if 'أنسة' in demo_text:
                            marital_status = 'أنسة'
                        elif 'متزوج' in demo_text:
                            marital_status = 'متزوجة' if 'متزوجة' in demo_text else 'متزوج'
                        elif 'أعزب' in demo_text:
                            marital_status = 'عزباء' if 'عزباء' in demo_text else 'أعزب'

    # Process remaining fields with full OCR if needed
    if not all([back_nid, job, education, religion, marital_status]):
        print("\n=== Running supplementary OCR for missing fields ===")
        reader_normal = get_reader('ar')
        try:
            gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
            ocr_results_normal = reader_normal.readtext(gray)
            
            for result in ocr_results_normal:
                if len(result) >= 2:
                    text = result[1] if isinstance(result[1], str) else str(result[1])
                    text = text.strip()
                    
                    if not job and 'طالب' in text:
                        job = text
                    elif not education and ('كلية' in text or 'جامعة' in text):
                        education = text
                    elif not religion and ('مسلم' in text or 'مسيحي' in text):
                        if 'مسلم' in text:
                            religion = 'مسلمة' if 'مسلمة' in text else 'مسلم'
                    elif not marital_status and ('أنسة' in text or 'متزوج' in text or 'أعزب' in text):
                        if 'أنسة' in text:
                            marital_status = 'أنسة'
        except Exception as e:
            print(f"Error during supplementary OCR: {e}")

    print("\n=== Final Values ===")
    print(f"Back NID: {back_nid}")
    print(f"Date Numbers (Left to Right): {date_numbers}")
    if date_mm_yyyy:
        print(f"Formatted Date (MM/YYYY): {date_mm_yyyy}")
        print(f"Formatted Date (YYYY/MM): {date_yyyy_mm}")
    print(f"Issue Date: {issue_date}")
    print(f"Expiry Date: {expiry_date}")
    print(f"Job: {job}")
    print(f"Education: {education}")
    print(f"Religion: {religion}")
    print(f"Marital Status: {marital_status}")

    return back_nid, issue_date, expiry_date, job, education, religion, marital_status

# print(detect_and_process_id_card("font_ID.jpg"))

def detect_and_process_back_side(image_path):
    """
    Detect and process the back side of an Egyptian ID card following the same flow as front side.
    
    Args:
        image_path (str): Path to the image file
        
    Returns:
        tuple: (back_nid, issue_date, expiry_date, job, education, religion, marital_status)
    """
    # Perform inference to detect the ID card
    id_card_results = id_card_model(image_path)
    print(f"ID Card Detection Results: {id_card_results}")

    # Load the original image using OpenCV
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load image")

    # Crop the ID card from the image
    cropped_image = None
    for result in id_card_results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            cropped_image = image[y1:y2, x1:x2]
            break  # Take the first detected card
        if cropped_image is not None:
            break

    # If no card was detected, use the full image
    if cropped_image is None:
        print("No ID card detected, using full image...")
        cropped_image = image

    # Pass the cropped image to the existing processing function
    return process_back_image(cropped_image)
