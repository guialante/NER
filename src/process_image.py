"""
Process images using OCR and extract text.
"""
import os
from ocr import OCRProcessor
from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_sample_image():
    """Create a sample image with text for OCR processing."""
    # Create a new image with white background
    width = 800
    height = 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Sample text
    text = """
    LEGAL DOCUMENT SAMPLE
    
    John Smith appointed Jane Doe as the trustee 
    of his estate on March 15, 2024.
    
    The trust agreement allocates $500,000 to 
    the beneficiary Sarah Williams.
    """
    
    # Wrap text to fit image width
    wrapper = textwrap.TextWrapper(width=40)
    wrapped_text = wrapper.fill(text)
    
    # Add text to image
    draw.text((50, 50), wrapped_text, fill='black')
    
    # Save the image
    image_path = "sample_document.png"
    image.save(image_path)
    return image_path

def main():
    # Create sample image
    print("\nCreating sample document image...")
    image_path = create_sample_image()
    print(f"Sample image created: {image_path}")
    
    # Initialize OCR processor
    processor = OCRProcessor()
    
    # Get image information
    print("\nImage Information:")
    print("=" * 50)
    info = processor.get_image_info(image_path)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Process image with default settings
    print("\nExtracted Text (Default Settings):")
    print("=" * 50)
    text = processor.process_image(image_path)
    print(text.strip())
    
    # Process image with custom settings
    print("\nExtracted Text (Custom Settings):")
    print("=" * 50)
    custom_config = '--oem 3 --psm 6'  # Use LSTM engine with uniform text block
    text = processor.process_image_with_config(image_path, custom_config)
    print(text.strip())
    
    # Clean up
    os.remove(image_path)
    print(f"\nCleaned up: Removed {image_path}")

if __name__ == "__main__":
    main() 