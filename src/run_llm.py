"""
Simple script to demonstrate LLM integration.
"""
from llm import LLMProcessor

def main():
    """Run LLM entity extraction and document generation."""
    # Sample text
    text = """
    John Smith appointed Jane Doe as the trustee of his estate on March 15, 2024.
    The trust agreement allocates $500,000 to the beneficiary Sarah Williams.
    Upon the death of Thomas Anderson, the successor trustee Elizabeth Wilson 
    shall distribute $2.5 million according to Article 7.
    """
    
    print("\nAnalyzing text with LLM...")
    print("=" * 50)
    print(text.strip())
    
    # Initialize LLM processor
    processor = LLMProcessor()
    
    # Extract entities from text
    print("\nExtracting entities...")
    try:
        entities = processor.extract_entities(text)
        if not entities:
            raise ValueError("No entities found")
    except Exception as e:
        print(f"Could not extract entities: {e}")
        print("Using default entities instead.")
        # Provide sample entities to proceed with the demo
        entities = [
            {"entity": "John Smith", "type": "Name"},
            {"entity": "Jane Doe", "type": "Name"},
            {"entity": "March 15, 2024", "type": "Date"},
            {"entity": "$500,000", "type": "MonetaryAmount"},
            {"entity": "Sarah Williams", "type": "Name"},
            {"entity": "Thomas Anderson", "type": "Name"},
            {"entity": "Elizabeth Wilson", "type": "Name"},
            {"entity": "$2.5 million", "type": "MonetaryAmount"},
            {"entity": "Article 7", "type": "LegalClause"},
            {"entity": "trustee", "type": "LegalTerm"},
            {"entity": "beneficiary", "type": "LegalTerm"},
            {"entity": "successor trustee", "type": "LegalTerm"}
        ]
    
    # Print extracted entities
    print("\nExtracted Entities:")
    print("=" * 50)
    for entity in entities:
        print(f"Entity: {entity.get('entity')}, Type: {entity.get('type')}")
    
    # Generate document from entities
    print("\nGenerating legal document...")
    document = processor.generate_document(entities, "Trust Agreement")
    
    # Save document in both formats
    txt_path = processor.save_txt_document(document, "generated_trust.txt")
    docx_path = processor.save_docx_document(document, "generated_trust.docx")
    
    print("\nProcess complete!")
    print(f"Text document saved to: {txt_path}")
    print(f"DOCX document saved to: {docx_path}")

if __name__ == "__main__":
    main() 