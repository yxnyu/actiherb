"""
Advanced example demonstrating different active learning strategies and configurations.
Uses separate CSV files for real-world scenarios.
"""

import pandas as pd
import matplotlib.pyplot as plt
from molecular_active_learning import MolecularActiveLearning
from molecular_active_learning.utils import load_data_from_files, setup_logging

def compare_strategies():
    """Compare different active learning strategies."""
    
    # Setup logging
    setup_logging('advanced_example.log')
    
    print("=== Advanced Molecular Active Learning Example ===")
    
    # Load data from separate CSV files
    initial_labeled, unlabeled_pool, oracle_data = load_data_from_files(
        initial_labeled_path='examples/data/initial_labeled.csv',
        unlabeled_pool_path='examples/data/unlabeled_pool.csv',
        oracle_data_path='examples/data/oracle_data.csv'
    )
    
    print(f"Data loaded:")
    print(f"- Initial labeled: {len(initial_labeled)} molecules")
    print(f"- Unlabeled pool: {len(unlabeled_pool)} molecules")
    print(f"- Oracle data: {len(oracle_data)} molecules")
    
    # Define strategies to compare
    strategies = ['BALD', 'RandomSampling', 'BatchBALD']
    results = {}
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy} strategy ---")
        
        # Create a copy of data for each strategy
        initial_copy = initial_labeled.copy()
        unlabeled_copy = unlabeled_pool.copy()
        
        # Initialize active learning with different strategy
        al = MolecularActiveLearning(
            initial_labeled_data=initial_copy,
            unlabeled_pool=unlabeled_copy,
            oracle_data=oracle_data,
            batch_size=2,
            max_iterations=3,
            strategy=strategy
        )
        
        # Run active learning
        strategy_results = al.run_active_learning()
        results[strategy] = strategy_results
        
        print(f"{strategy} completed: {len(strategy_results)} iterations")
    
    return results

def detailed_configuration_example():
    """Example with detailed configuration options."""
    
    print("\n=== Detailed Configuration Example ===")
    
    # Load data
    initial_labeled, unlabeled_pool, oracle_data = load_data_from_files(
        initial_labeled_path='examples/data/initial_labeled.csv',
        unlabeled_pool_path='examples/data/unlabeled_pool.csv',
        oracle_data_path='examples/data/oracle_data.csv'
    )
    
    # Advanced configuration
    al = MolecularActiveLearning(
        initial_labeled_data=initial_labeled,
        unlabeled_pool=unlabeled_pool,
        oracle_data=oracle_data,
        batch_size=5,           # Larger batch size
        max_iterations=10,      # More iterations
        strategy='BALD',        # Bayesian Active Learning by Disagreement
        model_type='unimol',    # UniMol model
        epochs=20,              # More training epochs
        learning_rate=0.001,    # Custom learning rate
        uncertainty_threshold=0.1,  # Stop if uncertainty drops below this
        save_models=True,       # Save intermediate models
        verbose=True            # Detailed logging
    )
    
    print("Starting detailed active learning run...")
    results = al.run_active_learning()
    
    # Analyze results
    print(f"\nDetailed Results:")
    print(f"- Iterations completed: {len(results)}")
    print(f"- Total molecules labeled: {len(al.labeled_data)}")
    print(f"- Efficiency: {len(al.labeled_data)/len(oracle_data)*100:.1f}% of dataset explored")
    
    return results

def visualize_results(results):
    """Visualize active learning results."""
    
    print("\n=== Visualization ===")
    
    # Plot strategy comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Cumulative molecules labeled
    for strategy, strategy_results in results.items():
        iterations = list(range(1, len(strategy_results) + 1))
        cumulative_labeled = []
        total_labeled = len(strategy_results[0]['selected_molecules']) if strategy_results else 0
        
        for i, result in enumerate(strategy_results):
            total_labeled += len(result['selected_molecules'])
            cumulative_labeled.append(total_labeled)
        
        ax1.plot(iterations, cumulative_labeled, marker='o', label=strategy)
    
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cumulative Labeled Molecules')
    ax1.set_title('Cumulative Labeling Progress')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Molecules per iteration
    for strategy, strategy_results in results.items():
        iterations = list(range(1, len(strategy_results) + 1))
        molecules_per_iter = [len(result['selected_molecules']) for result in strategy_results]
        
        ax2.bar([i + 0.2 * list(results.keys()).index(strategy) for i in iterations], 
                molecules_per_iter, width=0.2, label=strategy, alpha=0.7)
    
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Molecules Selected')
    ax2.set_title('Molecules Selected per Iteration')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('advanced_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Results visualization saved as 'advanced_results.png'")

def main():
    """Main function to run all advanced examples."""
    
    # Compare different strategies
    strategy_results = compare_strategies()
    
    # Run detailed configuration example
    detailed_results = detailed_configuration_example()
    
    # Visualize results
    visualize_results(strategy_results)
    
    print("\n🎉 Advanced examples completed!")
    print("Check the log files and visualization for detailed results.")

if __name__ == '__main__':
    main() 