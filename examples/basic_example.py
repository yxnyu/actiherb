"""
Basic example of molecular active learning using separate CSV files.
This demonstrates the standard workflow with real-world data inputs.
"""

import pandas as pd
from molecular_active_learning import MolecularActiveLearning
from molecular_active_learning.utils import load_data_from_files, setup_logging

def main():
    # Setup logging
    setup_logging('basic_example.log')
    
    print("=== Molecular Active Learning - Basic Example ===")
    
    # Load data from separate CSV files (real-world scenario)
    initial_labeled, unlabeled_pool, oracle_data = load_data_from_files(
        initial_labeled_path='examples/data/initial_labeled.csv',
        unlabeled_pool_path='examples/data/unlabeled_pool.csv', 
        oracle_data_path='examples/data/oracle_data.csv'
    )
    
    print(f"Data loaded:")
    print(f"- Initial labeled: {len(initial_labeled)} molecules")
    print(f"- Unlabeled pool: {len(unlabeled_pool)} molecules")
    print(f"- Oracle data: {len(oracle_data)} molecules")
    
    # Initialize active learning
    al = MolecularActiveLearning(
        initial_labeled_data=initial_labeled,
        unlabeled_pool=unlabeled_pool,
        oracle_data=oracle_data,
        batch_size=3,
        max_iterations=5
    )
    
    # Run active learning
    print("\nStarting active learning...")
    results = al.run_active_learning()
    
    # Display results
    print(f"\nActive Learning Results:")
    print(f"- Total iterations: {len(results)}")
    print(f"- Final labeled set size: {len(al.labeled_data)}")
    print(f"- Remaining unlabeled: {len(al.unlabeled_pool)}")
    
    # Show performance over iterations
    print(f"\nPerformance over iterations:")
    for i, result in enumerate(results):
        print(f"Iteration {i+1}: {result['selected_molecules']} molecules selected")

if __name__ == '__main__':
    main() 