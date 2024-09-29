mport streamlit as st
import pytesseract
from PIL import Image
from byaldi import RAGMultiModalModel
import cv2
import numpy as np

# Load the Byaldi model
RAG = RAGMultiModalModel.from_pretrained("vidore/colpali-v1.2")

# Define the Tesseract command path if needed
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

def perform_ocr(image):
    """Extract text from the uploaded image."""
    try:
        extracted_text = pytesseract.image_to_string(image, lang="eng+hin")
        if not extracted_text.strip():
            return "No text found in the image. Make sure Hindi language is supported."
        return extracted_text
    except Exception as e:
        return f"Error in OCR processing: {str(e)}"

def index_image(image):
    """Index the image using Byaldi."""
    try:
        image.save("uploaded_image.png")
        index_name = 'shubh_image_index'
        RAG.index(
            input_path="uploaded_image.png",
            index_name=index_name,
            store_collection_with_index=False,
            overwrite=True
        )
        return index_name
    except Exception as e:
        return f"Error in indexing: {str(e)}"

def search_keyword(extracted_text, keyword):
    """Search for the keyword in the extracted text."""
    if extracted_text and keyword:
        if keyword.lower() in extracted_text.lower():
            return f"Keyword '{keyword}' found in the extracted text!"
        else:
            return f"Keyword '{keyword}' not found in the extracted text."
    else:
        return "No text available to search."

def process_image(image):
    """Process the image, extract text, and index it."""
    extracted_text = perform_ocr(image)

    if "Error" in extracted_text or extracted_text == "No text found in the image.":
        return extracted_text, None

    index_result = index_image(image)
    if "Error" in index_result:
        return extracted_text, index_result

    return extracted_text, "Image indexed successfully."

# Streamlit interface
st.title("OCR and Keyword Search Application")

# Upload an image
uploaded_image = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

# If an image is uploaded, process it
if uploaded_image:
    image = Image.open(uploaded_image)
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