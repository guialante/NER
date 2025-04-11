"""
Evaluation metrics and test dataset handling for NER system.
"""
from typing import List, Dict, Tuple
from sklearn.metrics import precision_score, recall_score, f1_score
import json
import os

class NEREvaluator:
    def __init__(self):
        self.test_data = []
        self.results = {}

    def load_test_dataset(self, file_path: str) -> None:
        """
        Load test dataset from a JSON file.
        
        Format:
        [
            {
                "text": "Sample legal text",
                "entities": [
                    {"entity": "John Doe", "type": "Name"},
                    {"entity": "January 1, 2024", "type": "Date"}
                ]
            }
        ]
        """
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                self.test_data = json.load(f)

    def save_test_dataset(self, file_path: str) -> None:
        """Save test dataset to a JSON file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(self.test_data, f, indent=2)

    def add_test_case(self, text: str, entities: List[Dict[str, str]]) -> None:
        """Add a test case to the dataset."""
        self.test_data.append({
            "text": text,
            "entities": entities
        })

    def evaluate_predictions(self, 
                           true_entities: List[Dict[str, str]], 
                           pred_entities: List[Dict[str, str]]) -> Dict[str, float]:
        """
        Calculate precision, recall, and F1 score for entity predictions.
        
        Args:
            true_entities: List of ground truth entities
            pred_entities: List of predicted entities
            
        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        # Convert entities to binary labels for each entity type
        entity_types = set(e["type"] for e in true_entities + pred_entities)
        
        results = {}
        for entity_type in entity_types:
            # Create binary labels for this entity type
            y_true = []
            y_pred = []
            
            # Get all unique entity texts
            all_entities = set(e["entity"] for e in true_entities + pred_entities)
            
            for entity in all_entities:
                # Check if entity exists in true and predicted with correct type
                true_exists = any(e["entity"] == entity and e["type"] == entity_type 
                                for e in true_entities)
                pred_exists = any(e["entity"] == entity and e["type"] == entity_type 
                                for e in pred_entities)
                
                y_true.append(1 if true_exists else 0)
                y_pred.append(1 if pred_exists else 0)
            
            if len(y_true) > 0:  # Only calculate metrics if we have entities
                results[entity_type] = {
                    "precision": precision_score(y_true, y_pred, zero_division=0),
                    "recall": recall_score(y_true, y_pred, zero_division=0),
                    "f1": f1_score(y_true, y_pred, zero_division=0)
                }
        
        # Calculate overall metrics
        overall_true = []
        overall_pred = []
        all_entities = set(e["entity"] for e in true_entities + pred_entities)
        
        for entity in all_entities:
            true_exists = any(e["entity"] == entity for e in true_entities)
            pred_exists = any(e["entity"] == entity for e in pred_entities)
            overall_true.append(1 if true_exists else 0)
            overall_pred.append(1 if pred_exists else 0)
        
        results["overall"] = {
            "precision": precision_score(overall_true, overall_pred, zero_division=0),
            "recall": recall_score(overall_true, overall_pred, zero_division=0),
            "f1": f1_score(overall_true, overall_pred, zero_division=0)
        }
        
        return results

    def evaluate_model(self, prediction_function) -> Dict[str, Dict[str, float]]:
        """
        Evaluate model performance on the test dataset.
        
        Args:
            prediction_function: Function that takes text and returns list of entities
            
        Returns:
            Dictionary with evaluation metrics per entity type and overall
        """
        all_results = {}
        
        for test_case in self.test_data:
            text = test_case["text"]
            true_entities = test_case["entities"]
            
            # Get predictions
            pred_entities = prediction_function(text)
            
            # Calculate metrics
            metrics = self.evaluate_predictions(true_entities, pred_entities)
            
            # Aggregate results
            for entity_type, scores in metrics.items():
                if entity_type not in all_results:
                    all_results[entity_type] = {
                        "precision": [],
                        "recall": [],
                        "f1": []
                    }
                for metric, value in scores.items():
                    all_results[entity_type][metric].append(value)
        
        # Calculate average metrics
        final_results = {}
        for entity_type, scores in all_results.items():
            final_results[entity_type] = {
                metric: sum(values) / len(values)
                for metric, values in scores.items()
            }
        
        self.results = final_results
        return final_results

    def print_results(self) -> None:
        """Print evaluation results in a formatted way."""
        if not self.results:
            print("No evaluation results available.")
            return
        
        print("\nNER Evaluation Results")
        print("=" * 60)
        
        for entity_type, metrics in self.results.items():
            print(f"\nEntity Type: {entity_type}")
            print("-" * 30)
            for metric, value in metrics.items():
                print(f"{metric.capitalize():>10}: {value:.4f}") 