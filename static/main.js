// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // DOM Elements
    const dragArea = document.getElementById('dragArea');
    const fileInput = document.getElementById('fileInput');
    const browseBtn = document.getElementById('browseBtn');
    const previewArea = document.getElementById('previewArea');
    const previewImage = document.getElementById('previewImage');
    const removePreview = document.getElementById('removePreview');
    const frontBtn = document.getElementById('frontBtn');
    const backBtn = document.getElementById('backBtn');
    const loadingIndicator = document.getElementById('loadingIndicator');
    const resultsContent = document.getElementById('resultsContent');
    const processedImageArea = document.getElementById('processedImageArea');
    const processedImage = document.getElementById('processedImage');

// State
let currentSide = 'front';
let currentFile = null;

// Event Listeners
frontBtn.addEventListener('click', () => switchSide('front'));
backBtn.addEventListener('click', () => switchSide('back'));
browseBtn.addEventListener('click', () => fileInput.click());
removePreview.addEventListener('click', resetUpload);
fileInput.addEventListener('change', handleFileSelect);

// Drag and drop events
['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
    dragArea.addEventListener(eventName, preventDefaults, false);
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

['dragenter', 'dragover'].forEach(eventName => {
    dragArea.addEventListener(eventName, () => {
        dragArea.classList.add('active');
    });
});

['dragleave', 'drop'].forEach(eventName => {
    dragArea.addEventListener(eventName, () => {
        dragArea.classList.remove('active');
    });
});

dragArea.addEventListener('drop', handleDrop, false);

// Functions
function switchSide(side) {
    currentSide = side;
    if (side === 'front') {
        frontBtn.classList.add('bg-blue-600', 'text-white');
        frontBtn.classList.remove('bg-gray-200', 'text-gray-700');
        backBtn.classList.add('bg-gray-200', 'text-gray-700');
        backBtn.classList.remove('bg-blue-600', 'text-white');
    } else {
        backBtn.classList.add('bg-blue-600', 'text-white');
        backBtn.classList.remove('bg-gray-200', 'text-gray-700');
        frontBtn.classList.add('bg-gray-200', 'text-gray-700');
        frontBtn.classList.remove('bg-blue-600', 'text-white');
    }
    if (currentFile) {
        processIDCard(currentFile);
    }
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const files = dt.files;
    handleFiles(files);
}

function handleFileSelect(e) {
    const files = e.target.files;
    handleFiles(files);
}

function handleFiles(files) {
    if (files.length === 0) return;
    
    const file = files[0];
    if (!file.type.startsWith('image/')) {
        showToast('Please upload an image file', 'error');
        return;
    }

    currentFile = file;
    displayPreview(file);
    processIDCard(file);
}

function displayPreview(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewArea.classList.remove('hidden');
        dragArea.classList.add('hidden');
    };
    reader.readAsDataURL(file);
}

function resetUpload() {
    currentFile = null;
    fileInput.value = '';
    previewArea.classList.add('hidden');
    dragArea.classList.remove('hidden');
    resultsContent.innerHTML = `
        <div class="text-center text-gray-500 py-8">
            <i class="fas fa-database text-4xl mb-2"></i>
            <p>Upload an ID card to see extracted information</p>
        </div>
    `;
    processedImageArea.classList.add('hidden');
}

async function processIDCard(file) {
    try {
        loadingIndicator.classList.remove('hidden');
        resultsContent.innerHTML = '<div class="text-center text-gray-500 py-8">Processing...</div>';

        const formData = new FormData();
        formData.append('file', file);

        const endpoint = currentSide === 'front' ? '/api/v1/ocr/front' : '/api/v1/ocr/back';
        const response = await fetch(`${API_BASE_URL}${endpoint}`, {
            method: 'POST',
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.detail || 'Failed to process image');
        }

        displayResults(result);
        
        if (result.processed_image_url) {
            processedImage.src = `${API_BASE_URL}${result.processed_image_url}`;
            processedImageArea.classList.remove('hidden');
        }

        showToast('Processing completed successfully', 'success');
    } catch (error) {
        showToast(error.message, 'error');
        resultsContent.innerHTML = `
            <div class="text-center text-red-500 py-8">
                <i class="fas fa-exclamation-circle text-4xl mb-2"></i>
                <p>${error.message}</p>
            </div>
        `;
    } finally {
        loadingIndicator.classList.add('hidden');
    }
}

function displayResults(result) {
    if (!result.data) return;

    const data = result.data;
    let html = '<div class="space-y-3">';

    for (const [key, value] of Object.entries(data)) {
        const formattedKey = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
        const isArabic = /[\u0600-\u06FF]/.test(value);
        const textAlignment = isArabic ? 'rtl' : 'ltr';
        
        html += `
            <div class="bg-gray-50 p-3 rounded-lg">
                <div class="text-sm text-gray-500">${formattedKey}</div>
                <div class="text-lg font-medium" dir="${textAlignment}">${value || '-'}</div>
            </div>
        `;
    }

    html += '</div>';
    resultsContent.innerHTML = html;
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `animate-fade-in-up p-4 rounded-lg shadow-lg text-white ${
        type === 'error' ? 'bg-red-500' : 'bg-green-500'
    }`;
    toast.textContent = message;

    const toastContainer = document.getElementById('toastContainer');
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('opacity-0');
        setTimeout(() => {
            toastContainer.removeChild(toast);
        }, 300);
    }, 3000);

}
});