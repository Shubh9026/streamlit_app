import streamlit as st
import easyocr
from PIL import Image
from byaldi import RAGMultiModalModel  # Import Byaldi library
import numpy as np
import traceback
import os
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

# Load the Byaldi model
@st.cache_resource
def load_rag_model():
    try:
        return RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")
    except Exception as e:
        st.error(f"Error loading RAG model: {str(e)}")
        return None

reader = load_ocr_reader()
rag_model = load_rag_model()

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

def index_image(image):
    """Index the image using Byaldi."""
    if rag_model is None:
        return "RAG model not initialized properly."
    try:
        temp_image_path = "temp_uploaded_image.png"
        image.save(temp_image_path)
        index_name = 'shubh_image_index'
        rag_model.index(
            input_path=temp_image_path,
            index_name=index_name,
            store_collection_with_index=False,
            overwrite=True
        )
        os.remove(temp_image_path)  # Clean up the temporary file
        return index_name
    except Exception as e:
        st.error(f"Error in indexing: {str(e)}")
        st.error(traceback.format_exc())
        return None

def search_keyword_in_text(extracted_text, keyword):
    """Search for the keyword in the extracted text."""
    if extracted_text and keyword:
        return f"Keyword '{keyword}' {'found' if keyword.lower() in extracted_text.lower() else 'not found'} in the extracted text."
    return "No text available to search."

def search_image_index(keyword):
    """Search for the keyword in the indexed images using Byaldi."""
    if rag_model is None:
        return "RAG model not initialized properly."
    try:
        results = rag_model.search(query=keyword, index_name='shubh_image_index')
        return results
    except Exception as e:
        st.error(f"Error in search: {str(e)}")
        st.error(traceback.format_exc())
        return None

def process_image(image):
    """Process the image, extract text, and index it."""
    extracted_text = perform_ocr(image)
    index_result = index_image(image) if extracted_text else None
    return extracted_text, index_result

# Streamlit interface
st.title("OCR and Keyword Search Application")

uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        extracted_text, index_status = process_image(image)

        if extracted_text:
            st.subheader("Extracted Text")
            st.text(extracted_text)

        st.subheader("Indexing Status")
        st.text("Image indexed successfully." if index_status else "Indexing failed.")

        keyword = st.text_input("Enter a keyword to search")
        if keyword:
            search_result_text = search_keyword_in_text(extracted_text, keyword)
            st.subheader("Search Result in Text")
            st.text(search_result_text)

            image_search_results = search_image_index(keyword)
            st.subheader("Search Result in Image Index")
            st.text(image_search_results if image_search_results else "No results found in image index.")
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        st.error(traceback.format_exc())
