import streamlit as st
import pytesseract
from PIL import Image
from byaldi import RAGMultiModalModel
import cv2
import numpy as np
import os

# Load the Byaldi model
@st.cache_resource
def load_model():
    return RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

RAG = load_model()

# Define the Tesseract command path if needed
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

@st.cache_data
def preprocess_image(image):
    # Convert PIL Image to OpenCV format
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(thresh)

def perform_ocr(image):
    """Extract text from the uploaded image."""
    try:
        preprocessed_image = preprocess_image(image)
        extracted_text = pytesseract.image_to_string(preprocessed_image, lang="eng+hin")
        if not extracted_text.strip():
            return "No text found in the image. Make sure Hindi language is supported."
        return extracted_text
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        return None

def index_image(image):
    """Index the image using Byaldi."""
    try:
        temp_image_path = "temp_uploaded_image.png"
        image.save(temp_image_path)
        index_name = 'shubh_image_index'
        RAG.index(
            input_path=temp_image_path,
            index_name=index_name,
            store_collection_with_index=False,
            overwrite=True
        )
        os.remove(temp_image_path)  # Clean up the temporary file
        return index_name
    except Exception as e:
        st.error(f"Error in indexing: {str(e)}")
        return None

def search_keyword(extracted_text, keyword):
    """Search for the keyword in the extracted text."""
    if extracted_text and keyword:
        if keyword.lower() in extracted_text.lower():
            return f"Keyword '{keyword}' found in the extracted text!"
        else:
            return f"Keyword '{keyword}' not found in the extracted text."
    else:
        return "No text available to search."

@st.cache_data
def process_image(image):
    """Process the image, extract text, and index it."""
    extracted_text = perform_ocr(image)
    if not extracted_text:
        return None, None
    index_result = index_image(image)
    if not index_result:
        return extracted_text, None
    return extracted_text, "Image indexed successfully."

# Streamlit interface
st.title("OCR and Keyword Search Application")

# Upload an image
uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# If an image is uploaded, process it
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    extracted_text, index_status = process_image(image)

    # Display extracted text
    if extracted_text:
        st.subheader("Extracted Text")
        st.text(extracted_text)

    # Display index status
    if index_status:
        st.subheader("Indexing Status")
        st.text(index_status)

    # Keyword search
    keyword = st.text_input("Enter a keyword to search")
    if keyword:
        search_result = search_keyword(extracted_text, keyword)
        st.subheader("Search Result")
        st.text(search_result)