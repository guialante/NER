"""
Test script for evaluating NER model performance.
"""
import pytest
from src.ner import extract_entities
from .evaluation import NEREvaluator
import os

@pytest.fixture
def evaluator():
    """Fixture to create an NEREvaluator instance."""
    return NEREvaluator()

@pytest.fixture
def test_dataset_path():
    """Fixture providing the path to the test dataset."""
    return os.path.join('tests', 'data', 'test_dataset.json')

def test_load_dataset(evaluator, test_dataset_path):
    """Test loading the test dataset."""
    evaluator.load_test_dataset(test_dataset_path)
    assert len(evaluator.test_data) > 0
    assert all(isinstance(item, dict) for item in evaluator.test_data)
    assert all('text' in item and 'entities' in item for item in evaluator.test_data)

def test_evaluate_model(evaluator, test_dataset_path):
    """Test model evaluation on the test dataset."""
    evaluator.load_test_dataset(test_dataset_path)
    results = evaluator.evaluate_model(extract_entities)
    
    # Check that we have results for each entity type
    assert 'overall' in results
    assert all(metric in results['overall'] for metric in ['precision', 'recall', 'f1'])
    
    # Check that metrics are in valid range [0, 1]
    for entity_type, metrics in results.items():
        for metric_name, value in metrics.items():
            assert 0 <= value <= 1, f"Invalid {metric_name} for {entity_type}: {value}"

def test_individual_predictions(evaluator, test_dataset_path):
    """Test evaluation of individual predictions."""
    # Sample ground truth and predictions
    true_entities = [
        {"entity": "John Smith", "type": "Name"},
        {"entity": "March 15, 2024", "type": "Date"}
    ]
    pred_entities = [
        {"entity": "John Smith", "type": "Name"},
        {"entity": "March 15", "type": "Date"}
    ]
    
    results = evaluator.evaluate_predictions(true_entities, pred_entities)
    assert 'Name' in results
    assert 'Date' in results
    assert 'overall' in results

def test_empty_predictions(evaluator):
    """Test evaluation with empty predictions."""
    results = evaluator.evaluate_predictions([], [])
    assert results['overall']['precision'] == 0
    assert results['overall']['recall'] == 0
    assert results['overall']['f1'] == 0

if __name__ == '__main__':
    # Create evaluator
    evaluator = NEREvaluator()
    
    # Load test dataset
    test_dataset_path = os.path.join('tests', 'data', 'test_dataset.json')
    evaluator.load_test_dataset(test_dataset_path)
    
    # Evaluate model
    results = evaluator.evaluate_model(extract_entities)
    
    # Print results
    evaluator.print_results() 