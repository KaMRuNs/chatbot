from langchain_core.tools import tool
import pytesseract
from PIL import Image
import os

@tool
def extract_text_from_image(image_path: str) -> str:
    """
    Extract text computationally from an image file (OCR).
    Use this tool when the user uploads an image (PNG, JPG, JPEG) or asks to read text from a picture.
    Pass the absolute file path of the image.
    """
    if not os.path.exists(image_path):
        return f"Error: The image file at {image_path} does not exist."
    
    try:
        # On Windows, explicitly point to the Tesseract executable path we just installed
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        
        img = Image.open(image_path)
        extracted_text = pytesseract.image_to_string(img)
        
        if not extracted_text.strip():
            return "No text could be found or extracted from the image."
            
        return f"Extracted Text:\n\n{extracted_text}"
    except Exception as e:
        return f"An error occurred while trying to read the image: {str(e)}"
