"""
Named Entity Recognition module for legal documents.
"""
import spacy
import re
from datetime import datetime


def extract_entities(text):
    """
    Extract named entities from text using spaCy.
    
    Parameters:
    -----------
    text : str
        The input text to extract entities from.
        
    Returns:
    --------
    list of dict
        A list of dictionaries where each dictionary represents an entity.
        Format: {"entity": "John Doe", "type": "Name"}
    """
    # Load the English model
    nlp = spacy.load("en_core_web_lg")
    
    # Process the text
    doc = nlp(text)
    
    # Initialize entity list and tracking sets to avoid duplicates
    entities = []
    processed_spans = set()
    
    # Process spaCy's built-in entities
    for ent in doc.ents:
        # Create a unique identifier for this span
        span_id = (ent.start_char, ent.end_char)
        
        # Skip if we've already processed this span
        if span_id in processed_spans:
            continue
        
        # Add to processed spans
        processed_spans.add(span_id)
        
        # Classify entity based on spaCy's entity type
        if ent.label_ == "PERSON":
            entities.append({"entity": ent.text, "type": "Name"})
        elif ent.label_ == "DATE":
            entities.append({"entity": ent.text, "type": "Date"})
        elif ent.label_ == "MONEY":
            entities.append({"entity": ent.text, "type": "MonetaryAmount"})
    
    # Process legal terms using regex patterns
    legal_terms = [
        "trustee", "beneficiary", "executor", "grantor", "testator", 
        "settlor", "fiduciary", "heir", "legatee", "probate", "bequest",
        "devise", "legacy", "will", "trust", "estate", "power of attorney"
    ]
    
    # Find legal terms in text
    for term in legal_terms:
        pattern = re.compile(r'\b' + term + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            # Check if this span overlaps with any processed span
            overlap = False
            span = (match.start(), match.end())
            
            for proc_span in processed_spans:
                # Check for overlap
                if not (span[1] <= proc_span[0] or span[0] >= proc_span[1]):
                    overlap = True
                    break
            
            if not overlap:
                entities.append({"entity": match.group(), "type": "LegalTerm"})
                processed_spans.add(span)
    
    return entities


if __name__ == "__main__":
    # Simple test
    test_text = "John Doe signed the trust agreement on January 15, 2023, " \
                "allocating $50,000 to the beneficiary, Jane Smith."
    
    result = extract_entities(test_text)
    for entity in result:
        print(f"{entity['entity']} ({entity['type']})") 