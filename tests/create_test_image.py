from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    # Create a white background image
    width = 800
    height = 400
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Try to load a system font, fallback to default if not available
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
    except:
        font = ImageFont.load_default()
    
    # Sample text with various entity types
    text = """LEGAL DOCUMENT SAMPLE

Client: John Smith
Date: January 15, 2024
Amount: $50,000.00

Trust Agreement between John Smith (Trustor) and Sarah Johnson (Trustee)
Beneficiary: Emma Smith

Article 1.2: The trustee shall manage the trust assets
in accordance with the terms specified herein."""
    
    # Add text to image
    draw.text((50, 50), text, fill='black', font=font)
    
    # Create test_images directory if it doesn't exist
    os.makedirs('test_images', exist_ok=True)
    
    # Save the image
    image_path = 'test_images/test_document.png'
    image.save(image_path)
    print(f"Test image created at: {image_path}")
    return image_path

if __name__ == "__main__":
    create_test_image() 