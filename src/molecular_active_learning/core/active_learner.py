import pandas as pd
import numpy as np
import os
import logging
import traceback
from baal.active import ActiveLearningLoop
from baal.active.heuristics import BALD
from rdkit import Chem
from rdkit.Chem import Descriptors

from .dataset import MoleculeALDataset
from .model_wrapper import train_unimol_model
from ..utils.helpers import setup_logging, oracle_function


class MolecularActiveLearning:
    """Main class for molecular active learning."""
    
    def __init__(self, task='regression', epochs=10, batch_size=16, query_size=5,
                 metrics='pearsonr', model_name='unimolv2', model_size='1.1B',
                 log_file='active_learning.log'):
        """
        Initialize the molecular active learning system.
        
        Args:
            task: Task type ('regression' or 'classification')
            epochs: Number of training epochs per iteration
            batch_size: Training batch size
            query_size: Number of samples to query per iteration
            metrics: Evaluation metric
            model_name: UniMol model name
            model_size: Model size
            log_file: Path to log file
        """
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.query_size = query_size
        self.metrics = metrics
        self.model_name = model_name
        self.model_size = model_size
        
        # Setup logging
        setup_logging(log_file)
        
    def run(self, initial_labeled, unlabeled_pool, oracle_data, iterations=10, 
            save_intermediate=True, output_dir='./'):
        """
        Run the active learning loop.
        
        Args:
            initial_labeled: DataFrame with initial labeled data (SMILES, TARGET)
            unlabeled_pool: DataFrame with unlabeled data (SMILES)
            oracle_data: Complete dataset for oracle queries (SMILES, TARGET)
            iterations: Number of active learning iterations
            save_intermediate: Whether to save intermediate results
            output_dir: Directory for output files
            
        Returns:
            Final dataset with all labels
        """
        dataset = MoleculeALDataset(initial_labeled, unlabeled_pool)
        
        logging.info(f"Starting active learning with {len(initial_labeled)} labeled samples")
        logging.info(f"Unlabeled pool size: {len(unlabeled_pool)}")
        
        for i in range(iterations):
            logging.info(f"======== Iteration {i+1}/{iterations} ========")
            logging.info(f"Label status: {int(sum(dataset.labelled))}/{len(dataset.labelled)}")
            
            current_train_df = dataset.get_labeled_data()
            unlabeled_data = dataset.get_unlabeled_data()
            
            logging.info(f"Current training data size: {len(current_train_df)}")
            logging.info(f"Current unlabeled data size: {len(unlabeled_data)}")
            
            # Save current training data
            train_file = os.path.join(output_dir, f'train_round_{i+1}.csv')
            current_train_df.to_csv(train_file, index=False)
            
            if save_intermediate:
                current_train_df.to_csv(os.path.join(output_dir, f'labeled_round_{i+1}.csv'), index=False)
                unlabeled_data.to_csv(os.path.join(output_dir, f'unlabeled_round_{i+1}.csv'), index=False)
            
            # Create model directory
            model_dir = os.path.join(output_dir, f'model_round_{i+1}')
            
            try:
                # Train model
                model_wrapper = train_unimol_model(
                    train_file=train_file,
                    model_dir=model_dir,
                    task=self.task,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    metrics=self.metrics,
                    model_name=self.model_name,
                    model_size=self.model_size
                )
            except Exception as e:
                logging.error(f"Model training/loading error: {type(e).__name__}: {str(e)}")
                logging.error(traceback.format_exc())
                raise
            
            # Get unlabeled pool
            pool = dataset.pool
            predictions = model_wrapper.predict_on_dataset(pool)
            
            # Select samples based on uncertainty
            uncertainty_scores = predictions[:, 1, 0]
            query_indices = np.argsort(-uncertainty_scores)[:self.query_size]
            
            # Log uncertainty scores
            for idx, score in enumerate(uncertainty_scores):
                logging.info(f"Unlabeled sample {idx}, uncertainty score: {score:.4f}")
            
            logging.info(f"Selected sample indices: {query_indices}")
            logging.info(f"Corresponding uncertainty scores: {uncertainty_scores[query_indices]}")
            
            # Query oracle for new labels
            new_labels = []
            for idx in query_indices:
                smiles = pool[idx][0]
                uncertainty = predictions[idx, 1, 0]
                
                # Calculate molecular weight
                mol = Chem.MolFromSmiles(smiles)
                mol_wt = Descriptors.MolWt(mol) if mol else None
                
                logging.info(f"Selected sample {idx}: SMILES={smiles}, "
                           f"uncertainty={uncertainty:.4f}, mol_weight={mol_wt}")
                
                # Get true label from oracle
                label = oracle_function(smiles, oracle_data)
                new_labels.append(label)
            
            # Update dataset with new labels
            dataset.label(query_indices, new_labels)
            
            # Check if all samples are labeled
            if len(dataset.get_unlabeled_data()) == 0:
                final_labeled = dataset.get_labeled_data()
                final_labeled.to_csv(os.path.join(output_dir, 'final_labeled.csv'), index=False)
                logging.info("All samples labeled. Active learning terminated.")
                break
        
        return dataset 