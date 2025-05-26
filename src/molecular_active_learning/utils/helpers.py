import logging
import random
import pandas as pd


def setup_logging(log_file='active_learning.log'):
    """Setup logging configuration."""
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def oracle_function(smiles, oracle_data):
    """
    Oracle function to get true labels from the complete dataset.
    
    Args:
        smiles: SMILES string to query
        oracle_data: Complete dataset with SMILES and TARGET columns
        
    Returns:
        True target value for the given SMILES
    """
    matching_row = oracle_data[oracle_data['SMILES'] == smiles]
    if not matching_row.empty:
        return matching_row['TARGET'].values[0]
    else:
        logging.warning(f"SMILES not found in oracle data: {smiles}, returning random value")
        return random.uniform(0, 1)


def validate_input_data(initial_labeled, unlabeled_pool, oracle_data):
    """
    Validate input data for molecular active learning.
    
    Args:
        initial_labeled: DataFrame with SMILES and TARGET columns
        unlabeled_pool: DataFrame with SMILES column
        oracle_data: DataFrame with SMILES and TARGET columns (ground truth)
        
    Raises:
        AssertionError: If data validation fails
    """
    # Check required columns
    assert 'SMILES' in initial_labeled.columns, "initial_labeled must have 'SMILES' column"
    assert 'TARGET' in initial_labeled.columns, "initial_labeled must have 'TARGET' column"
    assert 'SMILES' in unlabeled_pool.columns, "unlabeled_pool must have 'SMILES' column"
    assert 'SMILES' in oracle_data.columns, "oracle_data must have 'SMILES' column"
    assert 'TARGET' in oracle_data.columns, "oracle_data must have 'TARGET' column"
    
    # Check for non-empty data
    assert len(initial_labeled) > 0, "initial_labeled cannot be empty"
    assert len(unlabeled_pool) > 0, "unlabeled_pool cannot be empty"
    assert len(oracle_data) > 0, "oracle_data cannot be empty"
    
    # Check for duplicates between labeled and unlabeled
    labeled_smiles = set(initial_labeled['SMILES'])
    unlabeled_smiles = set(unlabeled_pool['SMILES'])
    overlap = labeled_smiles.intersection(unlabeled_smiles)
    assert len(overlap) == 0, f"Duplicate SMILES found between labeled and unlabeled: {overlap}"
    
    # Check oracle coverage
    all_molecules = labeled_smiles.union(unlabeled_smiles)
    oracle_molecules = set(oracle_data['SMILES'])
    missing_in_oracle = all_molecules - oracle_molecules
    assert len(missing_in_oracle) == 0, f"Oracle missing molecules: {missing_in_oracle}"
    
    logging.info("Input data validation passed")


def load_data_from_files(initial_labeled_path, unlabeled_pool_path, oracle_data_path):
    """
    Load and validate data from separate CSV files.
    
    Args:
        initial_labeled_path: Path to CSV with initial labeled data (SMILES, TARGET)
        unlabeled_pool_path: Path to CSV with unlabeled molecules (SMILES)
        oracle_data_path: Path to CSV with complete ground truth data (SMILES, TARGET)
        
    Returns:
        Tuple of (initial_labeled_df, unlabeled_pool_df, oracle_data_df)
    """
    # Load data
    initial_labeled = pd.read_csv(initial_labeled_path)
    unlabeled_pool = pd.read_csv(unlabeled_pool_path)
    oracle_data = pd.read_csv(oracle_data_path)
    
    # Validate data
    validate_input_data(initial_labeled, unlabeled_pool, oracle_data)
    
    logging.info(f"Loaded data: {len(initial_labeled)} labeled, {len(unlabeled_pool)} unlabeled, {len(oracle_data)} oracle")
    
    return initial_labeled, unlabeled_pool, oracle_data 