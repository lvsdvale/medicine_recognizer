"""this implements fixtures"""

import os
import sys

import pytest
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_DIR)


@pytest.fixture
def mock_medication_image():
    """returns a PIL image of medication"""
    return Image.open("datasets/medicine_database/ibuprofeno/ibuprofeno_1.jpg")


@pytest.fixture
def mock_medication_image_path():
    """returns path to image of medication"""
    return "datasets/medicine_database/ibuprofeno/ibuprofeno_1.jpg"


@pytest.fixture
def sample_raw_text():
    """Fixture providing a sample raw OCR text."""
    return "This is an example text with some common words."


@pytest.fixture
def sample_processed_text():
    """Fixture providing a sample cleaned OCR."""
    return "example text common words"
