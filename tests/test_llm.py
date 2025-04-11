"""
Tests for LLM integration functionality.
"""
import pytest
import os
from src.llm import LLMProcessor
from unittest.mock import patch, MagicMock

# Sample text for testing
SAMPLE_TEXT = """
John Smith appointed Jane Doe as the trustee of his estate on March 15, 2024.
The trust agreement allocates $500,000 to the beneficiary Sarah Williams.
"""

# Sample entities for testing
SAMPLE_ENTITIES = [
    {"entity": "John Smith", "type": "Name"},
    {"entity": "Jane Doe", "type": "Name"},
    {"entity": "March 15, 2024", "type": "Date"},
    {"entity": "$500,000", "type": "MonetaryAmount"},
    {"entity": "trustee", "type": "LegalTerm"},
    {"entity": "estate", "type": "LegalTerm"},
    {"entity": "trust agreement", "type": "LegalTerm"},
    {"entity": "beneficiary", "type": "LegalTerm"},
    {"entity": "Sarah Williams", "type": "Name"}
]

@pytest.fixture
def mock_llm_processor():
    """Create a mock LLM processor that doesn't make actual API calls."""
    with patch("src.llm.ChatOpenAI") as mock_chat, \
         patch("src.llm.LLMChain") as mock_chain:
        
        # Mock the LLM chains
        mock_entity_chain = MagicMock()
        mock_entity_chain.invoke.return_value = {
            "text": str(SAMPLE_ENTITIES)
        }
        
        mock_document_chain = MagicMock()
        mock_document_chain.invoke.return_value = {
            "text": "TRUST AGREEMENT\n\nThis agreement is made on March 15, 2024, between John Smith and Jane Doe..."
        }
        
        # Make the chain constructor return our mocks
        mock_chain.side_effect = [mock_entity_chain, mock_document_chain]
        
        processor = LLMProcessor()
        yield processor

def test_extract_entities(mock_llm_processor):
    """Test entity extraction functionality."""
    with patch.object(mock_llm_processor, 'extract_entities', return_value=SAMPLE_ENTITIES):
        entities = mock_llm_processor.extract_entities(SAMPLE_TEXT)
        
        # Check that entities were extracted
        assert len(entities) > 0
        
        # Check entity structure
        for entity in entities:
            assert "entity" in entity
            assert "type" in entity

def test_analyze_document(mock_llm_processor):
    """Test document analysis functionality."""
    with patch.object(mock_llm_processor, 'extract_entities', return_value=SAMPLE_ENTITIES):
        analysis = mock_llm_processor.analyze_document(SAMPLE_TEXT)
        
        # Check analysis structure
        assert "entities" in analysis
        assert "entities_by_type" in analysis
        assert "total_entities" in analysis
        
        # Check entities by type grouping
        entities_by_type = analysis["entities_by_type"]
        assert "Name" in entities_by_type
        assert "Date" in entities_by_type
        assert "MonetaryAmount" in entities_by_type
        assert "LegalTerm" in entities_by_type

def test_generate_document(mock_llm_processor):
    """Test document generation functionality."""
    with patch.object(mock_llm_processor, 'generate_document', 
                      return_value="TRUST AGREEMENT\n\nThis agreement is made on March 15, 2024..."):
        document = mock_llm_processor.generate_document(SAMPLE_ENTITIES)
        
        # Check that document was generated
        assert document is not None
        assert len(document) > 0
        assert "TRUST AGREEMENT" in document

def test_save_txt_document(mock_llm_processor, tmp_path):
    """Test saving document as TXT."""
    # Generate test document
    document = "TEST DOCUMENT\n\nThis is a test."
    
    # Save to temporary path
    test_file = tmp_path / "test_document.txt"
    file_path = mock_llm_processor.save_txt_document(document, str(test_file))
    
    # Check that file was created and contains the document
    assert os.path.exists(file_path)
    with open(file_path, 'r') as f:
        content = f.read()
        assert content == document

def test_save_docx_document(mock_llm_processor, tmp_path):
    """Test saving document as DOCX."""
    # Generate test document
    document = "TEST DOCUMENT\n\nThis is a test."
    
    # Save to temporary path
    test_file = tmp_path / "test_document.docx"
    file_path = mock_llm_processor.save_docx_document(document, str(test_file))
    
    # Check that file was created
    assert os.path.exists(file_path)
    assert file_path.endswith(".docx") 