"""
Analyze legal documents using OCR and NER.
"""
import os
from ocr import OCRProcessor
from ner import extract_entities
from PIL import Image, ImageDraw, ImageFont
import textwrap

def create_sample_document():
    """Create a sample legal document image."""
    # Create a new image with white background
    width = 800
    height = 600
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Sample legal document text
    text = """
    LEGAL DOCUMENT ANALYSIS
    
    TRUST AGREEMENT
    
    This agreement, made on March 15, 2024, between:
    
    John Smith (hereinafter referred to as the "Grantor")
    and
    Jane Doe (hereinafter referred to as the "Trustee")
    
    The Grantor hereby transfers $500,000 to the Trustee
    for the benefit of Sarah Williams (hereinafter referred
    to as the "Beneficiary").
    
    Upon the death of Thomas Anderson, the successor trustee
    Elizabeth Wilson shall distribute $2.5 million according
    to Article 7 of this agreement.
    
    Signed this day,
    John Smith
    """
    
    # Wrap text to fit image width
    wrapper = textwrap.TextWrapper(width=40)
    wrapped_text = wrapper.fill(text)
    
    # Add text to image
    draw.text((50, 50), wrapped_text, fill='black')
    
    # Save the image
    image_path = "legal_document.png"
    image.save(image_path)
    return image_path

def main():
    # Create sample document
    print("\nCreating sample legal document...")
    document_path = create_sample_document()
    print(f"Sample document created: {document_path}")
    
    # Initialize OCR processor
    processor = OCRProcessor()
    
    # Get document information
    print("\nDocument Information:")
    print("=" * 50)
    info = processor.get_image_info(document_path)
    for key, value in info.items():
        print(f"{key}: {value}")
    
    # Extract text using OCR
    print("\nExtracted Text:")
    print("=" * 50)
    text = processor.process_image(document_path)
    print(text.strip())
    
    # Extract entities from the text
    print("\nExtracted Entities:")
    print("=" * 50)
    entities = extract_entities(text)
    
    # Group entities by type
    entities_by_type = {}
    for entity in entities:
        entity_type = entity['type']
        if entity_type not in entities_by_type:
            entities_by_type[entity_type] = []
        entities_by_type[entity_type].append(entity['entity'])
    
    # Print entities grouped by type
    for entity_type, entities_list in entities_by_type.items():
        print(f"\n{entity_type}:")
        for entity in entities_list:
            print(f"  - {entity}")
    
    # Clean up
    os.remove(document_path)
    print(f"\nCleaned up: Removed {document_path}")

if __name__ == "__main__":
    main() 