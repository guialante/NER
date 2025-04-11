# Legal Document NER System

A comprehensive Named Entity Recognition (NER) system designed for legal documents, combining OCR capabilities with advanced NLP for accurate entity extraction.

## Project Structure

```
NER/
├── src/
│   ├── __init__.py
│   ├── ner.py
│   ├── custom_ner.py
│   ├── combined_ner.py
│   ├── ocr.py
│   ├── llm.py
│   ├── evaluation.py
│   ├── extract_text.py
│   ├── process_image.py
│   ├── analyze_document.py
│   ├── analyze_with_llm.py
│   └── run_llm.py
├── tests/
│   ├── __init__.py
│   ├── test_ner.py
│   ├── test_ocr.py
│   ├── test_llm.py
│   └── test_ner_evaluation.py
├── pytest.ini
├── .env
├── .gitignore
└── requirements.txt
```

## Installation

1. Clone this repository:
```bash
git clone git@github.com:guialante/NER.git
cd NER
```

2. Create and activate a virtual environment:
```bash
python -m venv ner-app
source ner-app/bin/activate
```

3. Install the required Python packages:
```bash
pip install -r requirements.txt
```

4. Set up your OpenAI API key:
   - Create a `.env` file in the project root
   - Add your API key: `OPENAI_API_KEY=your_openai_api_key_here`

## Quick Start

The project includes several ready-to-use scripts that demonstrate the system's capabilities:

### 1. Text Extraction

Extract entities from legal text:

```bash
python src/extract_text.py
```

This script processes a sample legal document and extracts entities like names, dates, and monetary amounts.

### 2. Image Processing

Process images with OCR:

```bash
python src/process_image.py
```

This script creates a sample document image, processes it with OCR, and displays the extracted text.

### 3. Document Analysis

Analyze legal documents using both OCR and NER:

```bash
python src/analyze_document.py
```

This script demonstrates the full pipeline: creating a sample document, extracting text with OCR, and identifying entities with NER.

### 4. LLM Integration

Run the LLM-based entity extraction and document generation:

```bash
python src/run_llm.py
```

This simple script demonstrates:
- Extracting entities from legal text using OpenAI's GPT models
- Generating legal documents based on extracted entities
- Saving documents in both TXT and DOCX formats

### 5. Complete LLM-Powered Analysis

Analyze legal documents with the complete workflow:

```bash
python src/analyze_with_llm.py
```

This script showcases:
- Creating a sample document image
- Extracting text using OCR
- Identifying entities using LLM
- Generating formatted legal documents in DOCX and TXT formats

## Advanced Usage

### 1. Entity Extraction with LLM

```python
from src.llm import LLMProcessor

# Initialize LLM processor
processor = LLMProcessor()

# Example text
text = """
John Smith appointed Jane Doe as the trustee of his estate on March 15, 2024.
The trust agreement allocates $500,000 to the beneficiary Sarah Williams.
"""

# Extract entities
entities = processor.extract_entities(text)
for entity in entities:
    print(f"Entity: {entity['entity']}, Type: {entity['type']}")
```

### 2. Document Generation

```python
from src.llm import LLMProcessor

# Initialize LLM processor
processor = LLMProcessor()

# Generate document from entities
entities = [
    {"entity": "John Smith", "type": "Name"},
    {"entity": "March 15, 2024", "type": "Date"},
    {"entity": "$500,000", "type": "MonetaryAmount"}
]
document = processor.generate_document(entities, "Trust Agreement")
print(document)

# Save as DOCX
docx_path = processor.save_docx_document(document, "trust_agreement.docx")
print(f"DOCX created at: {docx_path}")

# Save as TXT
txt_path = processor.save_txt_document(document, "trust_agreement.txt")
print(f"TXT created at: {txt_path}")
```

## LLM Implementation Details

The project implements LLM integration using:

1. **OpenAI API**: Direct integration with OpenAI's GPT models for entity extraction and document generation
2. **Entity Extraction**: Uses prompt engineering to extract specific entity types from legal documents
3. **Document Generation**: Transforms extracted entities into professionally formatted legal documents
4. **Document Formatting**: Support for DOCX and TXT output with appropriate formatting

The LLM implementation includes:
- Structured prompt templates for consistent results
- JSON parsing of entity data
- Error handling to ensure robust operation
- Document styling for professional output

## Running Tests

The project uses pytest for testing. You can run the tests using:

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_ocr.py
pytest tests/test_ner.py
pytest tests/test_llm.py

# Run tests with detailed output
pytest -v

# Run tests and show print statements
pytest -s

# Run tests without warnings
pytest --disable-warnings
```

## Entity Types

The system can recognize the following entity types:
- **Names**: Person names, organization names
- **Dates**: Document dates, deadlines, timelines
- **MonetaryAmounts**: Financial figures, amounts
- **LegalTerms**: Legal roles, document types
- **LegalClauses**: References to specific sections or articles
- **Custom Entities**: Client information, assets, legal clauses
