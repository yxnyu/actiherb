import unittest
import pandas as pd
import numpy as np
from molecular_active_learning.core.dataset import MolDataset, MoleculeALDataset
from molecular_active_learning.utils import validate_input_data, load_data_from_files


class TestBasicFunctionality(unittest.TestCase):
    
    def setUp(self):
        # Create sample data
        self.labeled_df = pd.DataFrame({
            'SMILES': ['CCO', 'CCN', 'CCC'],
            'TARGET': [0.5, 0.7, 0.3]
        })
        
        self.unlabeled_df = pd.DataFrame({
            'SMILES': ['CCCO', 'CCCN', 'CCCC']
        })
    
    def test_mol_dataset_creation(self):
        data_dict = {0: 'CCO', 1: 'CCN'}
        label_dict = {0: 0.5, 1: 0.7}
        dataset = MolDataset(data_dict, label_dict)
        
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset[0], ('CCO', 0.5))
        self.assertEqual(dataset[1], ('CCN', 0.7))
    
    def test_molecule_al_dataset_creation(self):
        al_dataset = MoleculeALDataset(self.labeled_df, self.unlabeled_df)
        
        self.assertEqual(len(al_dataset), 6)  # 3 labeled + 3 unlabeled
        
        labeled_data = al_dataset.get_labeled_data()
        unlabeled_data = al_dataset.get_unlabeled_data()
        
        self.assertEqual(len(labeled_data), 3)
        self.assertEqual(len(unlabeled_data), 3)
    
    def test_labeling_process(self):
        al_dataset = MoleculeALDataset(self.labeled_df, self.unlabeled_df)
        
        # Initially 3 labeled, 3 unlabeled
        self.assertEqual(len(al_dataset.get_labeled_data()), 3)
        self.assertEqual(len(al_dataset.get_unlabeled_data()), 3)
        
        # Label one more sample
        al_dataset.label([3], [0.8])  # Label first unlabeled sample
        
        # Now should be 4 labeled, 2 unlabeled
        self.assertEqual(len(al_dataset.get_labeled_data()), 4)
        self.assertEqual(len(al_dataset.get_unlabeled_data()), 2)
    
    def test_validate_input_data(self):
        """Test input data validation function."""
        
        # Create valid test data
        initial_labeled = pd.DataFrame({
            'SMILES': ['CCO', 'CCN'],
            'TARGET': [0.1, 0.2]
        })
        
        unlabeled_pool = pd.DataFrame({
            'SMILES': ['CCC', 'CCCO']
        })
        
        oracle_data = pd.DataFrame({
            'SMILES': ['CCO', 'CCN', 'CCC', 'CCCO'],
            'TARGET': [0.1, 0.2, 0.3, 0.4]
        })
        
        # Should not raise any exception
        validate_input_data(initial_labeled, unlabeled_pool, oracle_data)
        
        # Test with duplicate SMILES
        bad_unlabeled = pd.DataFrame({
            'SMILES': ['CCO', 'CCCO']  # CCO already in labeled
        })
        
        with self.assertRaises(AssertionError):
            validate_input_data(initial_labeled, bad_unlabeled, oracle_data)
        
        # Test with missing oracle data
        incomplete_oracle = pd.DataFrame({
            'SMILES': ['CCO', 'CCN'],  # Missing CCC and CCCO
            'TARGET': [0.1, 0.2]
        })
        
        with self.assertRaises(AssertionError):
            validate_input_data(initial_labeled, unlabeled_pool, incomplete_oracle)


if __name__ == '__main__':
    unittest.main() 