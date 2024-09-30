import streamlit as st
import easyocr
from PIL import Image
import numpy as np
import traceback
import cv2

# Monkey patch to fix ANTIALIAS deprecation
Image.ANTIALIAS = Image.LANCZOS

# Initialize EasyOCR reader
@st.cache_resource
def load_ocr_reader():
    try:
        return easyocr.Reader(['en', 'hi'])  # For English and Hindi
    except Exception as e:
        st.error(f"Error loading OCR reader: {str(e)}")
        return None

reader = load_ocr_reader()

def perform_ocr(image):
    """Extract text from the uploaded image using EasyOCR."""
    if reader is None:
        return "OCR reader not initialized properly."
    try:
        # Convert PIL Image to numpy array
        img_array = np.array(image)
        # Convert RGB to BGR (if necessary)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        result = reader.readtext(img_array)
        extracted_text = ' '.join([text for _, text, _ in result])
        return extracted_text.strip() or "No text found in the image."
    except Exception as e:
        st.error(f"Error in OCR processing: {str(e)}")
        st.error(traceback.format_exc())
        return None

def search_keyword(extracted_text, keyword):
    """Search for the keyword in the extracted text."""
    if extracted_text and keyword:
        return f"Keyword '{keyword}' {'found' if keyword.lower() in extracted_text.lower() else 'not found'} in the extracted text."
    return "No text available to search."

def process_image(image):
    """Process the image and extract text."""
    extracted_text = perform_ocr(image)
    return extracted_text

# Streamlit interface
st.title("OCR and Keyword Search Application")

uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        extracted_text = process_image(image)

        if extracted_text:
            st.subheader("Extracted Text")
            st.text(extracted_text)

        keyword = st.text_input("Enter a keyword to search")
        if keyword:
            search_result = search_keyword(extracted_text, keyword)
            st.subheader("Search Result")
            st.text(search_result)
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        st.error(traceback.format_exc())