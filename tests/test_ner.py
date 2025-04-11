"""
Test script for Named Entity Recognition module.
"""
import pytest
from src.ner import extract_entities

@pytest.fixture
def test_cases():
    """Fixture providing test cases for NER."""
    return {
        "names_and_legal": {
            "text": "John Smith appointed Jane Doe as the trustee of his estate.",
            "expected_entities": [
                {"entity": "John Smith", "type": "Name"},
                {"entity": "Jane Doe", "type": "Name"},
                {"entity": "trustee", "type": "LegalTerm"},
                {"entity": "estate", "type": "LegalTerm"}
            ]
        },
        "dates_and_monetary": {
            "text": "The agreement was signed on June 15, 2022, with a payment of $250,000.",
            "expected_entities": [
                {"entity": "June 15, 2022", "type": "Date"},
                {"entity": "250,000", "type": "MonetaryAmount"}
            ]
        },
        "complex_legal": {
            "text": """The last will and testament of Michael Johnson, dated March 3, 2023, 
            bequeaths $75,000 to the beneficiary Sarah Williams. 
            The executor, Robert Thompson, shall distribute the assets by December 31, 2023.""",
            "expected_entities": [
                {"entity": "Michael Johnson", "type": "Name"},
                {"entity": "March 3", "type": "Date"},
                {"entity": "2023", "type": "Date"},
                {"entity": "75,000", "type": "MonetaryAmount"},
                {"entity": "Sarah Williams", "type": "Name"},
                {"entity": "Robert Thompson", "type": "Name"},
                {"entity": "December 31", "type": "Date"},
                {"entity": "2023", "type": "Date"},
                {"entity": "beneficiary", "type": "LegalTerm"},
                {"entity": "executor", "type": "LegalTerm"},
                {"entity": "will", "type": "LegalTerm"}
            ]
        }
    }

def test_entity_extraction(test_cases):
    """Test entity extraction for various test cases."""
    for case_name, case in test_cases.items():
        entities = extract_entities(case["text"])
        assert entities is not None
        assert isinstance(entities, list)
        
        # Check that all expected entities are present
        for expected in case["expected_entities"]:
            assert any(
                e["entity"] == expected["entity"] and e["type"] == expected["type"]
                for e in entities
            ), f"Expected entity {expected} not found in {case_name}"

def test_empty_text():
    """Test entity extraction with empty text."""
    entities = extract_entities("")
    assert entities == []

def test_no_entities():
    """Test entity extraction with text containing no entities."""
    text = "This is a simple sentence without any entities."
    entities = extract_entities(text)
    assert entities == []
 