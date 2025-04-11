import pytesseract
from PIL import Image
import os

class OCRProcessor:
    def __init__(self):
        """Initialize the OCR processor."""
        # Check if Tesseract is installed
        if not self._check_tesseract_installed():
            raise RuntimeError("Tesseract is not installed. Please install it first.")
    
    def _check_tesseract_installed(self):
        """Check if Tesseract is installed on the system."""
        try:
            pytesseract.get_tesseract_version()
            return True
        except pytesseract.TesseractNotFoundError:
            return False
    
    def process_image(self, image_path, lang='eng'):
        """
        Process an image and extract text using default settings.
        
        Args:
            image_path (str): Path to the image file
            lang (str): Language code for OCR (default: 'eng')
            
        Returns:
            str: Extracted text from the image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Extract text using default settings
            text = pytesseract.image_to_string(image, lang=lang)
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def process_image_with_config(self, image_path, config, lang='eng'):
        """
        Process an image with custom Tesseract configuration.
        
        Args:
            image_path (str): Path to the image file
            config (str): Tesseract configuration string
            lang (str): Language code for OCR (default: 'eng')
            
        Returns:
            str: Extracted text from the image
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            # Open the image
            image = Image.open(image_path)
            
            # Extract text with custom configuration
            text = pytesseract.image_to_string(image, lang=lang, config=config)
            return text.strip()
            
        except Exception as e:
            raise Exception(f"Error processing image: {str(e)}")
    
    def get_image_info(self, image_path):
        """
        Get information about the image.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Dictionary containing image information
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            return {
                'format': image.format,
                'mode': image.mode,
                'size': image.size,
                'width': image.width,
                'height': image.height
            }
        except Exception as e:
            raise Exception(f"Error getting image info: {str(e)}") 