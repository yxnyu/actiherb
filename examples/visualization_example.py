"""
Visualization example for molecular active learning results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import os


def analyze_learning_progress(output_dir='./example_output/'):
    """Analyze and visualize the active learning progress."""
    
    # Find all round files
    round_files = []
    for i in range(1, 50):  # Check up to 50 rounds
        file_path = os.path.join(output_dir, f'labeled_round_{i}.csv')
        if os.path.exists(file_path):
            round_files.append((i, file_path))
        else:
            break
    
    if not round_files:
        print(f"No round files found in {output_dir}")
        return
    
    # Collect data for each round
    rounds_data = []
    for round_num, file_path in round_files:
        df = pd.read_csv(file_path)
        rounds_data.append({
            'round': round_num,
            'n_samples': len(df),
            'target_mean': df['TARGET'].mean(),
            'target_std': df['TARGET'].std(),
            'target_min': df['TARGET'].min(),
            'target_max': df['TARGET'].max()
        })
    
    progress_df = pd.DataFrame(rounds_data)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Active Learning Progress Analysis', fontsize=16)
    
    # 1. Number of samples over rounds
    axes[0, 0].plot(progress_df['round'], progress_df['n_samples'], 'o-', linewidth=2, markersize=6)
    axes[0, 0].set_xlabel('Round')
    axes[0, 0].set_ylabel('Number of Labeled Samples')
    axes[0, 0].set_title('Dataset Growth')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Target distribution evolution
    axes[0, 1].plot(progress_df['round'], progress_df['target_mean'], 'o-', label='Mean', linewidth=2)
    axes[0, 1].fill_between(progress_df['round'], 
                           progress_df['target_mean'] - progress_df['target_std'],
                           progress_df['target_mean'] + progress_df['target_std'],
                           alpha=0.3, label='±1 Std')
    axes[0, 1].set_xlabel('Round')
    axes[0, 1].set_ylabel('Target Value')
    axes[0, 1].set_title('Target Distribution Evolution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Target range expansion
    axes[1, 0].plot(progress_df['round'], progress_df['target_min'], 'o-', label='Min', linewidth=2)
    axes[1, 0].plot(progress_df['round'], progress_df['target_max'], 'o-', label='Max', linewidth=2)
    axes[1, 0].fill_between(progress_df['round'], progress_df['target_min'], progress_df['target_max'], 
                           alpha=0.2, label='Range')
    axes[1, 0].set_xlabel('Round')
    axes[1, 0].set_ylabel('Target Value')
    axes[1, 0].set_title('Target Range Coverage')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Sampling efficiency (target std over rounds)
    axes[1, 1].plot(progress_df['round'], progress_df['target_std'], 'o-', linewidth=2, color='red')
    axes[1, 1].set_xlabel('Round')
    axes[1, 1].set_ylabel('Target Standard Deviation')
    axes[1, 1].set_title('Target Diversity')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_progress.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return progress_df


def analyze_molecular_properties(final_labeled_file):
    """Analyze molecular properties of the final labeled dataset."""
    
    if not os.path.exists(final_labeled_file):
        print(f"File not found: {final_labeled_file}")
        return
    
    df = pd.read_csv(final_labeled_file)
    
    # Calculate molecular descriptors
    descriptors = []
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            desc = {
                'SMILES': smiles,
                'MolWt': Descriptors.MolWt(mol),
                'LogP': Descriptors.MolLogP(mol),
                'HBD': Descriptors.NumHDonors(mol),
                'HBA': Descriptors.NumHAcceptors(mol),
                'TPSA': Descriptors.TPSA(mol),
                'NumRotBonds': Descriptors.NumRotatableBonds(mol),
                'NumRings': Descriptors.RingCount(mol)
            }
        else:
            desc = {col: np.nan for col in ['SMILES', 'MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 'NumRotBonds', 'NumRings']}
            desc['SMILES'] = smiles
        descriptors.append(desc)
    
    desc_df = pd.DataFrame(descriptors)
    
    # Merge with target values
    analysis_df = df.merge(desc_df, on='SMILES')
    
    # Create molecular property visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Molecular Properties Analysis', fontsize=16)
    
    properties = ['MolWt', 'LogP', 'HBD', 'HBA', 'TPSA', 'NumRotBonds']
    
    for i, prop in enumerate(properties):
        row, col = i // 3, i % 3
        
        # Scatter plot with target correlation
        scatter = axes[row, col].scatter(analysis_df[prop], analysis_df['TARGET'], 
                                       alpha=0.7, s=50, c=analysis_df['TARGET'], 
                                       cmap='viridis')
        axes[row, col].set_xlabel(prop)
        axes[row, col].set_ylabel('TARGET')
        axes[row, col].set_title(f'{prop} vs TARGET')
        
        # Add correlation coefficient
        corr = analysis_df[prop].corr(analysis_df['TARGET'])
        axes[row, col].text(0.05, 0.95, f'r = {corr:.3f}', 
                           transform=axes[row, col].transAxes, 
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=axes, shrink=0.6)
    cbar.set_label('TARGET Value')
    
    plt.savefig('molecular_properties_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary statistics
    print("\nMolecular Properties Summary:")
    print("="*50)
    print(analysis_df[properties + ['TARGET']].describe())
    
    return analysis_df


def create_selection_heatmap(output_dir='./example_output/'):
    """Create a heatmap showing when each sample was selected."""
    
    # Load all round data
    selection_data = []
    
    for i in range(1, 50):  # Check up to 50 rounds
        file_path = os.path.join(output_dir, f'labeled_round_{i}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                selection_data.append({
                    'SMILES': row['SMILES'],
                    'TARGET': row['TARGET'],
                    'selected_round': i
                })
        else:
            break
    
    if not selection_data:
        print("No selection data found")
        return
    
    selection_df = pd.DataFrame(selection_data)
    
    # Group by TARGET value ranges
    selection_df['target_bin'] = pd.cut(selection_df['TARGET'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    
    # Create pivot table for heatmap
    heatmap_data = selection_df.groupby(['selected_round', 'target_bin']).size().unstack(fill_value=0)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data.T, annot=True, fmt='d', cmap='Blues', 
                cbar_kws={'label': 'Number of Samples Selected'})
    plt.title('Sample Selection Pattern by Target Value Range')
    plt.xlabel('Selection Round')
    plt.ylabel('Target Value Range')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'selection_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    return selection_df


if __name__ == '__main__':
    # Example usage
    print("Analyzing active learning results...")
    
    # Analyze learning progress
    output_dir = './example_output/'
    if os.path.exists(output_dir):
        progress_data = analyze_learning_progress(output_dir)
        print("Learning progress analysis completed.")
        
        # Analyze final molecular properties
        final_file = os.path.join(output_dir, 'final_labeled.csv')
        if os.path.exists(final_file):
            mol_analysis = analyze_molecular_properties(final_file)
            print("Molecular properties analysis completed.")
        
        # Create selection heatmap
        selection_analysis = create_selection_heatmap(output_dir)
        print("Selection pattern analysis completed.")
    else:
        print(f"Output directory {output_dir} not found. Run basic_example.py first.") 