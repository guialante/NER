"""
Combined Named Entity Recognition module for legal documents.
This module combines standard and custom entity extraction.
"""
import spacy
import re
from pathlib import Path
from ner import extract_entities as extract_standard_entities
from custom_ner import extract_custom_entities, CUSTOM_ENTITIES


def extract_all_entities(text, custom_model_path):
    """
    Extract both standard and custom entities from text.
    
    Parameters:
    -----------
    text : str
        The input text to extract entities from.
    custom_model_path : str
        Path to the custom NER model.
        
    Returns:
    --------
    list of dict
        A list of dictionaries where each dictionary represents an entity.
        Format: {"entity": "John Doe", "type": "Name"} or {"entity": "John Doe", "type": "CLIENT"}
    """
    # Extract standard entities
    standard_entities = extract_standard_entities(text)
    
    # Extract custom entities if model exists
    custom_entities = []
    custom_model_path = Path(custom_model_path)
    
    if custom_model_path.exists():
        try:
            # Load the custom model
            custom_nlp = spacy.load(custom_model_path)
            
            # Extract custom entities
            custom_entities = extract_custom_entities(custom_nlp, text)
        except Exception as e:
            print(f"Error loading custom model: {e}")
    
    # Combine entities
    all_entities = standard_entities + custom_entities
    
    return all_entities


if __name__ == "__main__":
    # Test with a sample text
    test_text = """
    Client John Smith signed the trust agreement on January 15, 2023, 
    allocating $50,000 to the beneficiary, his daughter Sarah Smith.
    According to Article 3.2, the beach house shall be transferred to 
    the charitable foundation if the conditions in Paragraph 8 are met.
    """
    
    custom_model_path = "../models/custom_ner"
    
    # First check if we need to train the model
    if not Path(custom_model_path).exists():
        print("Custom model not found. Please run custom_ner.py first to train the model.")
    else:
        # Extract all entities
        entities = extract_all_entities(test_text, custom_model_path)
        
        # Group entities by type for better visualization
        entity_groups = {}
        for entity in entities:
            entity_type = entity["type"]
            if entity_type not in entity_groups:
                entity_groups[entity_type] = []
            entity_groups[entity_type].append(entity["entity"])
        
        # Print entities by type
        print("\nExtracted Entities:")
        print("-" * 50)
        for entity_type, entity_list in entity_groups.items():
            print(f"{entity_type}:")
            for entity_text in entity_list:
                print(f"  • {entity_text}")
            print() 