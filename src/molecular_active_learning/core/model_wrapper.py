import pandas as pd
import numpy as np
import os
import logging
from unimol_tools import MolTrain, MolPredict


class UnimolWrapper:
    """Wrapper for UniMol model to interface with BAAL."""
    
    def __init__(self, model_path, batch_size=16):
        self.model_path = model_path
        self.batch_size = batch_size
        
    def predict_on_dataset(self, dataset):
        """
        Make predictions on a dataset.
        
        Args:
            dataset: ActiveLearningPool object
            
        Returns:
            Array of shape (N, K, MC) where:
            N = number of samples
            K = number of classes (simulated as 2 for regression)
            MC = Monte Carlo samples (set to 1)
        """
        # Collect all SMILES
        smiles_list = []
        for i in range(len(dataset)):
            smiles, _ = dataset[i]
            smiles_list.append(smiles)
            
        # Create temporary DataFrame and save as CSV
        temp_df = pd.DataFrame({'SMILES': smiles_list})
        temp_file = 'temp_predict.csv'
        temp_df.to_csv(temp_file, index=False)
        
        # Check if model directory exists
        if not os.path.exists(self.model_path):
            logging.error(f"Model path does not exist: {self.model_path}")
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")
            
        # Check if config.yaml exists
        config_path = os.path.join(self.model_path, 'config.yaml')
        if not os.path.exists(config_path):
            logging.error(f"Config file does not exist: {config_path}")
            raise FileNotFoundError(f"Config file does not exist: {config_path}")
        
        # Use absolute path
        abs_model_path = os.path.abspath(self.model_path)
        logging.info(f"Using absolute model path: {abs_model_path} for prediction")
        
        # Make predictions using MolPredict
        predictor = MolPredict(load_model=abs_model_path)
        predictions = predictor.predict(data=temp_file)
        
        # Convert regression values to probability format expected by BAAL
        predictions = np.array(predictions)
        # Normalize to [0,1] range
        predictions = (predictions - predictions.min()) / (predictions.max() - predictions.min() + 1e-8)
        # Create binary classification probability distribution (1-p and p)
        probs = np.stack([1-predictions, predictions], axis=1)
        # Add MC dimension
        probs = probs.reshape(len(dataset), 2, 1)
        
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
            
        return probs


def train_unimol_model(train_file, model_dir, task='regression', epochs=10, batch_size=16, 
                      metrics='pearsonr', model_name='unimolv2', model_size='1.1B', kfold=1):
    """
    Train a UniMol model.
    
    Args:
        train_file: Path to training CSV file
        model_dir: Directory to save the trained model
        task: Task type ('regression' or 'classification')
        epochs: Number of training epochs
        batch_size: Training batch size
        metrics: Evaluation metric
        model_name: UniMol model name
        model_size: Model size
        kfold: K-fold cross validation
        
    Returns:
        Trained model wrapper
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    clf = MolTrain(
        task=task,
        data_type='molecule', 
        epochs=epochs,
        batch_size=batch_size,
        metrics=metrics,
        model_name=model_name,
        model_size=model_size,
        kfold=kfold,
        output_path=model_dir,
        save_path=model_dir
    )
    clf.fit(data=train_file)
    logging.info(f"Model trained successfully, saved in: {model_dir}")
    
    return UnimolWrapper(model_path=model_dir, batch_size=batch_size) 