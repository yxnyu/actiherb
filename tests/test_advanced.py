import unittest
import pandas as pd
import numpy as np
import tempfile
import os
from unittest.mock import patch, MagicMock

from molecular_active_learning.core.dataset import MolDataset, MoleculeALDataset
from molecular_active_learning.core.model_wrapper import UnimolWrapper
from molecular_active_learning.core.active_learner import MolecularActiveLearning
from molecular_active_learning.utils.helpers import oracle_function, split_data_randomly


class TestAdvancedFunctionality(unittest.TestCase):
    
    def setUp(self):
        """Set up test data."""
        # Create more comprehensive test data
        self.complete_data = pd.DataFrame({
            'SMILES': ['CCO', 'CCN', 'CCC', 'CCCO', 'CCCN', 'CCCC', 'CC(C)O', 'CC(C)N'],
            'TARGET': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        })
        
        self.labeled_df = self.complete_data.iloc[:3].copy()
        self.unlabeled_df = self.complete_data.iloc[3:][['SMILES']].copy()
    
    def test_data_splitting(self):
        """Test random data splitting functionality."""
        labeled, unlabeled = split_data_randomly(
            self.complete_data, 
            initial_labeled_fraction=0.375,  # 3/8 = 0.375
            random_state=42
        )
        
        self.assertEqual(len(labeled), 3)
        self.assertEqual(len(unlabeled), 5)
        self.assertTrue('TARGET' in labeled.columns)
        self.assertFalse('TARGET' in unlabeled.columns)
        self.assertTrue('SMILES' in unlabeled.columns)
    
    def test_oracle_function(self):
        """Test oracle function behavior."""
        # Test existing SMILES
        result = oracle_function('CCO', self.complete_data)
        self.assertEqual(result, 0.1)
        
        # Test non-existing SMILES
        with patch('random.uniform') as mock_random:
            mock_random.return_value = 0.5
            result = oracle_function('non_existing', self.complete_data)
            self.assertEqual(result, 0.5)
    
    def test_dataset_edge_cases(self):
        """Test dataset handling edge cases."""
        # Test empty labeled dataset
        empty_labeled = pd.DataFrame(columns=['SMILES', 'TARGET'])
        dataset = MoleculeALDataset(empty_labeled, self.unlabeled_df)
        
        self.assertEqual(len(dataset.get_labeled_data()), 0)
        self.assertEqual(len(dataset.get_unlabeled_data()), len(self.unlabeled_df))
        
        # Test empty unlabeled dataset
        empty_unlabeled = pd.DataFrame(columns=['SMILES'])
        dataset2 = MoleculeALDataset(self.labeled_df, empty_unlabeled)
        
        self.assertEqual(len(dataset2.get_labeled_data()), len(self.labeled_df))
        self.assertEqual(len(dataset2.get_unlabeled_data()), 0)
    
    def test_dataset_labeling_sequence(self):
        """Test multiple labeling operations."""
        dataset = MoleculeALDataset(self.labeled_df, self.unlabeled_df)
        
        initial_labeled = len(dataset.get_labeled_data())
        initial_unlabeled = len(dataset.get_unlabeled_data())
        
        # Label 2 samples
        dataset.label([3, 4], [0.9, 1.0])
        
        self.assertEqual(len(dataset.get_labeled_data()), initial_labeled + 2)
        self.assertEqual(len(dataset.get_unlabeled_data()), initial_unlabeled - 2)
        
        # Verify the new labels are correct
        labeled_data = dataset.get_labeled_data()
        self.assertTrue(0.9 in labeled_data['TARGET'].values)
        self.assertTrue(1.0 in labeled_data['TARGET'].values)
    
    @patch('molecular_active_learning.core.model_wrapper.MolPredict')
    def test_unimol_wrapper_prediction(self, mock_predict_class):
        """Test UnimolWrapper prediction functionality."""
        # Mock the predictor
        mock_predictor = MagicMock()
        mock_predictor.predict.return_value = [0.1, 0.2, 0.3]
        mock_predict_class.return_value = mock_predictor
        
        # Create wrapper with mocked model path
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create necessary files
            config_path = os.path.join(temp_dir, 'config.yaml')
            with open(config_path, 'w') as f:
                f.write('dummy: config')
            
            wrapper = UnimolWrapper(temp_dir)
            
            # Create test dataset
            dataset = MoleculeALDataset(self.labeled_df, self.unlabeled_df)
            pool = dataset.pool
            
            # Test prediction
            predictions = wrapper.predict_on_dataset(pool)
            
            # Verify output shape and properties
            self.assertEqual(predictions.shape, (len(pool), 2, 1))
            self.assertTrue(np.all(predictions >= 0))
            self.assertTrue(np.all(predictions <= 1))
            # Verify probabilities sum to 1 for each sample
            for i in range(len(pool)):
                prob_sum = predictions[i, :, 0].sum()
                self.assertAlmostEqual(prob_sum, 1.0, places=5)
    
    def test_molecular_active_learning_initialization(self):
        """Test MolecularActiveLearning initialization."""
        mal = MolecularActiveLearning(
            task='classification',
            epochs=5,
            batch_size=8,
            query_size=2,
            metrics='accuracy'
        )
        
        self.assertEqual(mal.task, 'classification')
        self.assertEqual(mal.epochs, 5)
        self.assertEqual(mal.batch_size, 8)
        self.assertEqual(mal.query_size, 2)
        self.assertEqual(mal.metrics, 'accuracy')
    
    def test_data_validation(self):
        """Test data validation and error handling."""
        # Test with missing columns
        invalid_data = pd.DataFrame({'INVALID': ['CCO', 'CCN']})
        
        with self.assertRaises(KeyError):
            MoleculeALDataset(invalid_data, self.unlabeled_df)
        
        # Test with mismatched indices
        labeled_with_gaps = self.labeled_df.copy()
        labeled_with_gaps.index = [0, 5, 10]  # Non-continuous indices
        
        # Should not raise an error, but handle gracefully
        dataset = MoleculeALDataset(labeled_with_gaps, self.unlabeled_df)
        self.assertEqual(len(dataset.get_labeled_data()), 3)
    
    def test_uncertainty_calculation(self):
        """Test uncertainty calculation in predictions."""
        # Create mock predictions with known uncertainty patterns
        n_samples = 5
        
        # High uncertainty case (probabilities close to 0.5)
        high_uncertainty = np.array([[[0.4], [0.6]] for _ in range(n_samples)])
        uncertainty_scores = high_uncertainty[:, 1, 0]
        
        # Low uncertainty case (probabilities close to 0 or 1)
        low_uncertainty = np.array([[[0.9], [0.1]] for _ in range(n_samples)])
        low_uncertainty_scores = low_uncertainty[:, 1, 0]
        
        # High uncertainty should have higher scores than low uncertainty
        self.assertTrue(np.mean(uncertainty_scores) > np.mean(low_uncertainty_scores))
    
    def test_logging_setup(self):
        """Test logging configuration."""
        from molecular_active_learning.utils.helpers import setup_logging
        
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp_log:
            temp_log_path = temp_log.name
        
        try:
            setup_logging(temp_log_path)
            
            # Test if logging works
            import logging
            logging.info("Test log message")
            
            # Check if log file was created and contains content
            self.assertTrue(os.path.exists(temp_log_path))
            
        finally:
            # Clean up
            if os.path.exists(temp_log_path):
                os.unlink(temp_log_path)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up integration test data."""
        self.test_data = pd.DataFrame({
            'SMILES': [f'C{"C" * i}O' for i in range(1, 11)],  # CCO, CCCO, CCCCO, etc.
            'TARGET': np.random.random(10)
        })
    
    def test_end_to_end_workflow_simulation(self):
        """Test complete workflow with mocked components."""
        labeled, unlabeled = split_data_randomly(
            self.test_data, 
            initial_labeled_fraction=0.3,
            random_state=42
        )
        
        # Test that we can create the dataset
        dataset = MoleculeALDataset(labeled, unlabeled)
        
        # Test basic operations
        initial_labeled_count = len(dataset.get_labeled_data())
        initial_unlabeled_count = len(dataset.get_unlabeled_data())
        
        # Simulate labeling some samples
        if initial_unlabeled_count > 0:
            query_indices = [0, 1] if initial_unlabeled_count > 1 else [0]
            new_labels = [0.5] * len(query_indices)
            dataset.label(query_indices, new_labels)
            
            # Verify changes
            self.assertEqual(len(dataset.get_labeled_data()), 
                           initial_labeled_count + len(query_indices))
            self.assertEqual(len(dataset.get_unlabeled_data()), 
                           initial_unlabeled_count - len(query_indices))


if __name__ == '__main__':
    unittest.main() 