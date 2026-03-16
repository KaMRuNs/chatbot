from langchain_core.tools import tool
import pytesseract
from PIL import Image, ImageOps, ImageEnhance
import os
import shutil


SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def _resolve_tesseract_cmd() -> str:
    """Resolve a working Tesseract executable path with sensible Windows fallbacks."""
    env_cmd = os.getenv("TESSERACT_CMD")
    if env_cmd and os.path.exists(env_cmd):
        return env_cmd

    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for path in common_paths:
        if os.path.exists(path):
            return path

    found = shutil.which("tesseract")
    if found:
        return found

    raise FileNotFoundError(
        "Tesseract executable was not found. Install Tesseract OCR or set TESSERACT_CMD in your .env file."
    )


def _prepare_image_for_ocr(image_path: str) -> Image.Image:
    """Open and preprocess an image to improve OCR readability."""
    ext = os.path.splitext(image_path)[1].lower()
    if ext not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {ext}. Supported formats: {', '.join(sorted(SUPPORTED_IMAGE_EXTENSIONS))}."
        )

    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    if image.width < 80 or image.height < 80:
        raise ValueError("Image is too small for reliable OCR. Use an image at least 80x80 pixels.")

    # Convert to grayscale and increase contrast for cleaner OCR input.
    gray = ImageOps.grayscale(image)
    enhanced = ImageEnhance.Contrast(gray).enhance(1.6)
    return enhanced


def _avg_confidence(conf_data: dict) -> float:
    values = []
    for value in conf_data.get("conf", []):
        try:
            score = float(value)
        except (TypeError, ValueError):
            continue
        if score >= 0:
            values.append(score)
    if not values:
        return 0.0
    return sum(values) / len(values)

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
        pytesseract.pytesseract.tesseract_cmd = _resolve_tesseract_cmd()
        img = _prepare_image_for_ocr(image_path)
        extracted_text = pytesseract.image_to_string(img)
        confidence_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confidence = _avg_confidence(confidence_data)
        
        if not extracted_text.strip():
            return (
                "No text could be found or extracted from the image. "
                "Try a higher-resolution image with clearer contrast."
            )
        warning = ""
        if confidence < 45:
            warning = (
                "\n\nWarning: OCR confidence is low. "
                "The image may be blurry, rotated, handwritten, or low contrast."
            )
            
        return f"Extracted Text (confidence: {confidence:.0f}%):\n\n{extracted_text.strip()}{warning}"
    except Exception as e:
        return f"An error occurred while trying to read the image: {str(e)}"
