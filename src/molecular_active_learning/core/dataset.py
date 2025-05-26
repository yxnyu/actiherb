import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from baal.active import ActiveLearningDataset


class MolDataset(Dataset):
    """Custom molecular dataset for handling SMILES and target data."""
    
    def __init__(self, data_dict, label_dict):
        self.data = data_dict
        self.labels = label_dict
        self.indices = list(data_dict.keys())
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        return self.data[real_idx], self.labels.get(real_idx, 0.0)


class MoleculeALDataset(ActiveLearningDataset):
    """Active learning dataset for molecular data."""
    
    def __init__(self, labeled_df, unlabeled_df):
        """
        Initialize the active learning dataset.
        
        Args:
            labeled_df: DataFrame with 'SMILES' and 'TARGET' columns
            unlabeled_df: DataFrame with 'SMILES' column only
        """
        # Build data dictionaries
        data_dict = {}
        label_dict = {}
        
        # Process labeled data
        for idx, row in labeled_df.iterrows():
            data_dict[idx] = row['SMILES']
            label_dict[idx] = row['TARGET']
            
        # Process unlabeled data
        offset = labeled_df.index.max() + 1 if not labeled_df.empty else 0
        for idx, row in unlabeled_df.iterrows():
            new_idx = idx + offset
            data_dict[new_idx] = row['SMILES']
        
        # Create base dataset
        self._dataset = MolDataset(data_dict, label_dict)
        
        # Initialize labeling status
        labelled = np.zeros(len(data_dict))
        for idx in label_dict.keys():
            labelled[list(data_dict.keys()).index(idx)] = 1
            
        # Initialize parent class
        super().__init__(
            dataset=self._dataset,
            labelled=labelled,
            make_unlabelled=lambda x: (x[0], 0.0)
        )
        
        # Store additional info for DataFrame format access
        self.data_dict = data_dict
        self.label_dict = label_dict
    
    def label(self, indices, new_labels):
        """Update labels for given indices."""
        super().label(indices, new_labels)
        for idx, label in zip(indices, new_labels):
            real_idx = self._dataset.indices[idx]
            self.label_dict[real_idx] = label
    
    def get_labeled_data(self):
        """Return labeled data as DataFrame."""
        labeled_indices = [i for i, l in enumerate(self.labelled) if l]
        smiles = []
        targets = []
        for i in labeled_indices:
            real_idx = self._dataset.indices[i]
            if real_idx in self.label_dict:
                smiles.append(self.data_dict[real_idx])
                targets.append(self.label_dict[real_idx])
        return pd.DataFrame({'SMILES': smiles, 'TARGET': targets})
    
    def get_unlabeled_data(self):
        """Return unlabeled data as DataFrame."""
        unlabeled_indices = [i for i, l in enumerate(self.labelled) if not l]
        smiles = []
        for i in unlabeled_indices:
            real_idx = self._dataset.indices[i]
            if real_idx in self.data_dict:
                smiles.append(self.data_dict[real_idx])
        return pd.DataFrame({'SMILES': smiles}) 