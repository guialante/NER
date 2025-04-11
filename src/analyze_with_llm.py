"""
Legal document analysis with LLM integration.
"""
import os
from llm import LLMProcessor
from ocr import OCRProcessor
from PIL import Image, ImageDraw
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
    image_path = "legal_document_sample.png"
    image.save(image_path)
    return image_path

def main():
    """Main function to process a document with LLM integration."""
    # Create sample document
    print("\nCreating sample legal document...")
    document_path = create_sample_document()
    print(f"Sample document created: {document_path}")
    
    # Initialize processors
    ocr_processor = OCRProcessor()
    llm_processor = LLMProcessor()
    
    # Extract text using OCR
    print("\nExtracting text with OCR...")
    text = ocr_processor.process_image(document_path)
    print("\nExtracted Text:")
    print("=" * 50)
    print(text.strip())
    
    # Extract entities using LLM
    print("\nExtracting entities with LLM...")
    analysis_result = llm_processor.analyze_document(text)
    
    # Print entities by type
    print("\nExtracted Entities:")
    print("=" * 50)
    for entity_type, entities in analysis_result["entities_by_type"].items():
        print(f"\n{entity_type}:")
        for entity in entities:
            print(f"  - {entity}")
    
    print(f"\nTotal entities extracted: {analysis_result['total_entities']}")
    
    # Generate documents
    print("\nGenerating formatted documents...")
    
    # Generate document text
    document_text = llm_processor.generate_document(analysis_result["entities"], "Trust Agreement")
    
    # Save as DOCX
    docx_path = llm_processor.save_docx_document(document_text, "trust_agreement.docx")
    if docx_path:
        print(f"\nDOCX document generated: {docx_path}")
    
    # Save as TXT
    txt_path = llm_processor.save_txt_document(document_text, "trust_agreement.txt")
    if txt_path:
        print(f"\nText document generated: {txt_path}")
    
    # Clean up sample document
    os.remove(document_path)
    print(f"\nCleaned up: Removed {document_path}")
    
    print("\nDocument analysis and generation complete!")

if __name__ == "__main__":
    main() 