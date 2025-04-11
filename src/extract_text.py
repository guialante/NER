"""
Text extraction from legal documents using NER.
"""
from ner import extract_entities

def main():
    # Sample legal document text
    legal_text = """
    John Smith appointed Jane Doe as the trustee of his estate on March 15, 2024.
    The trust agreement allocates $500,000 to the beneficiary Sarah Williams.
    Upon the death of Thomas Anderson, the successor trustee Elizabeth Wilson 
    shall distribute $2.5 million according to Article 7.
    """
    
    print("\nLegal Document Text:")
    print("=" * 50)
    print(legal_text.strip())
    
    print("\nExtracted Entities:")
    print("=" * 50)
    entities = extract_entities(legal_text)
    
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

if __name__ == "__main__":
    main() 