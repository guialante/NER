import pytest
from src.ocr import OCRProcessor
import os
from .create_test_image import create_test_image

@pytest.fixture
def ocr_processor():
    """Fixture to create an OCR processor instance."""
    return OCRProcessor()

@pytest.fixture
def test_image():
    """Fixture to create and return a test image path."""
    image_path = 'test_images/test_document.png'
    if not os.path.exists(image_path):
        image_path = create_test_image()
    return image_path

def test_process_image_default(ocr_processor, test_image):
    """Test image processing with default settings."""
    text = ocr_processor.process_image(test_image)
    assert text is not None
    assert isinstance(text, str)
    assert len(text) > 0
    assert "LEGAL DOCUMENT SAMPLE" in text
    assert "John Smith" in text

def test_process_image_custom_config(ocr_processor, test_image):
    """Test image processing with custom configuration."""
    custom_config = r'--oem 3 --psm 6'
    text = ocr_processor.process_image_with_config(test_image, custom_config)
    assert text is not None
    assert isinstance(text, str)
    assert len(text) > 0
    assert "LEGAL DOCUMENT SAMPLE" in text
    assert "John Smith" in text

def test_get_image_info(ocr_processor, test_image):
    """Test getting image information."""
    info = ocr_processor.get_image_info(test_image)
    assert info is not None
    assert isinstance(info, dict)
    assert info['format'] == 'PNG'
    assert info['mode'] == 'RGB'
    assert info['width'] == 800
    assert info['height'] == 400

def test_invalid_image_path(ocr_processor):
    """Test handling of invalid image path."""
    with pytest.raises(FileNotFoundError):
        ocr_processor.process_image("nonexistent_image.png") 