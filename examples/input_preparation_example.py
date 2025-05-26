"""
Input preparation examples for molecular active learning.
This file demonstrates how to prepare separate CSV files for real-world scenarios.
"""

import pandas as pd
import numpy as np
from molecular_active_learning.utils import validate_input_data

def example_1_drug_discovery_scenario():
    """
    Example 1: Drug discovery scenario
    You have some experimental data and want to expand your dataset
    """
    print("=== Example 1: Drug Discovery Scenario ===")
    
    # Scenario: You've already tested a few compounds in the lab
    initial_labeled = pd.DataFrame({
        'SMILES': [
            'CC(C)(C)c1ccc2occ(CC(=O)Nc3ccccc3F)c2c1',  # Known drug compound
            'COc1ccc(CCN)cc1',                           # Tested compound 1  
            'CC(=O)Nc1ccc(O)cc1'                        # Tested compound 2
        ],
        'TARGET': [0.75, 0.42, 0.58]  # Experimental activity values
    })
    
    # You have a large virtual library to screen
    unlabeled_pool = pd.DataFrame({
        'SMILES': [
            'CCN(CC)CCCC(C)Nc1ccnc2cc(Cl)ccc12',
            'CC1CCN(C(=O)c2ccc(F)cc2)CC1',
            'COc1ccc2nc(S(=O)(=O)N3CCN(C)CC3)sc2c1',
            'Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1',
            'CCOc1ccc2ncc(C#N)c(Nc3ccc(OCc4ccccn4)c(Cl)c3)c2c1'
        ]
    })
    
    # Oracle data contains ground truth for all compounds (simulated)
    all_smiles = list(initial_labeled['SMILES']) + list(unlabeled_pool['SMILES'])
    oracle_targets = list(initial_labeled['TARGET']) + [0.23, 0.67, 0.81, 0.35, 0.92]
    
    oracle_data = pd.DataFrame({
        'SMILES': all_smiles,
        'TARGET': oracle_targets
    })
    
    # Save files
    initial_labeled.to_csv('drug_discovery_initial_labeled.csv', index=False)
    unlabeled_pool.to_csv('drug_discovery_unlabeled_pool.csv', index=False)
    oracle_data.to_csv('drug_discovery_oracle_data.csv', index=False)
    
    print(f"✅ Drug discovery scenario created:")
    print(f"   - Initial labeled: {len(initial_labeled)} compounds")
    print(f"   - Unlabeled pool: {len(unlabeled_pool)} compounds")
    print(f"   - Oracle data: {len(oracle_data)} compounds")
    
    # Validate
    validate_input_data(initial_labeled, unlabeled_pool, oracle_data)
    print("   - Data validation: PASSED")
    
    return initial_labeled, unlabeled_pool, oracle_data

def example_2_material_science_scenario():
    """
    Example 2: Material science scenario  
    Screening for photovoltaic materials
    """
    print("\n=== Example 2: Material Science Scenario ===")
    
    # Initial experimental data from literature
    initial_labeled = pd.DataFrame({
        'SMILES': [
            'c1ccc2c(c1)oc1ccccc12',     # Dibenzofuran
            'c1ccc2sc3ccccc3c2c1',       # Dibenzothiophene
            'c1ccc2[nH]c3ccccc3c2c1'     # Carbazole
        ],
        'TARGET': [0.45, 0.67, 0.82]  # Power conversion efficiency
    })
    
    # Candidate materials to test
    unlabeled_pool = pd.DataFrame({
        'SMILES': [
            'c1ccc2c(c1)sc1ccccc12',     # Phenothiazine derivative
            'c1ccc2c(c1)nc1ccccc12',     # Phenanthroline
            'c1ccc2c(c1)cc1ccccc12',     # Phenanthrene  
            'c1ccc2c(c1)c1ccccc1n2',     # Acridine
            'c1ccc2c(c1)c1ccccc1o2',     # Dibenzopyran
            'c1ccc2c(c1)c1ccccc1s2'      # Dibenzothiophene variant
        ]
    })
    
    # Simulated ground truth
    all_smiles = list(initial_labeled['SMILES']) + list(unlabeled_pool['SMILES'])
    oracle_targets = list(initial_labeled['TARGET']) + [0.73, 0.56, 0.41, 0.68, 0.52, 0.71]
    
    oracle_data = pd.DataFrame({
        'SMILES': all_smiles,
        'TARGET': oracle_targets
    })
    
    # Save files
    initial_labeled.to_csv('materials_initial_labeled.csv', index=False)
    unlabeled_pool.to_csv('materials_unlabeled_pool.csv', index=False)
    oracle_data.to_csv('materials_oracle_data.csv', index=False)
    
    print(f"✅ Material science scenario created:")
    print(f"   - Initial labeled: {len(initial_labeled)} materials")
    print(f"   - Unlabeled pool: {len(unlabeled_pool)} materials")
    print(f"   - Oracle data: {len(oracle_data)} materials")
    
    # Validate
    validate_input_data(initial_labeled, unlabeled_pool, oracle_data)
    print("   - Data validation: PASSED")
    
    return initial_labeled, unlabeled_pool, oracle_data

def example_3_toxicity_prediction_scenario():
    """
    Example 3: Toxicity prediction scenario
    Safety assessment of industrial chemicals
    """
    print("\n=== Example 3: Toxicity Prediction Scenario ===")
    
    # Known toxic/non-toxic compounds from databases
    initial_labeled = pd.DataFrame({
        'SMILES': [
            'CCCCCCBr',          # 1-Bromohexane (toxic)
            'CCCCCCCO',          # 1-Heptanol (low toxicity)
            'c1ccc(Cl)cc1Cl',    # 1,4-Dichlorobenzene (toxic)
            'CCCCCCCC(=O)O'      # Octanoic acid (low toxicity)
        ],
        'TARGET': [0.85, 0.15, 0.92, 0.08]  # Toxicity scores (0=safe, 1=toxic)
    })
    
    # Chemicals requiring safety assessment
    unlabeled_pool = pd.DataFrame({
        'SMILES': [
            'CCCCCCCCC',         # Nonane
            'CCCCCCCCBr',        # 1-Bromooctane
            'c1ccc(Br)cc1',      # Bromobenzene
            'CCCCCCCCCO',        # 1-Nonanol
            'c1ccc(I)cc1',       # Iodobenzene
            'CCCCCCCCCC',        # Decane
            'c1ccc(F)cc1F',      # 1,4-Difluorobenzene
            'CCCCCCCCCCO'        # 1-Decanol
        ]
    })
    
    # Simulated ground truth for all compounds
    all_smiles = list(initial_labeled['SMILES']) + list(unlabeled_pool['SMILES'])
    oracle_targets = list(initial_labeled['TARGET']) + [0.05, 0.78, 0.65, 0.12, 0.72, 0.03, 0.35, 0.09]
    
    oracle_data = pd.DataFrame({
        'SMILES': all_smiles,
        'TARGET': oracle_targets
    })
    
    # Save files
    initial_labeled.to_csv('toxicity_initial_labeled.csv', index=False)
    unlabeled_pool.to_csv('toxicity_unlabeled_pool.csv', index=False)
    oracle_data.to_csv('toxicity_oracle_data.csv', index=False)
    
    print(f"✅ Toxicity prediction scenario created:")
    print(f"   - Initial labeled: {len(initial_labeled)} chemicals")
    print(f"   - Unlabeled pool: {len(unlabeled_pool)} chemicals")
    print(f"   - Oracle data: {len(oracle_data)} chemicals")
    
    # Validate
    validate_input_data(initial_labeled, unlabeled_pool, oracle_data)
    print("   - Data validation: PASSED")
    
    return initial_labeled, unlabeled_pool, oracle_data

def data_preparation_guidelines():
    """
    Provide guidelines for preparing your own data
    """
    print("\n=== Data Preparation Guidelines ===")
    
    guidelines = [
        "📋 **Required CSV Files:**",
        "   1. initial_labeled.csv - Molecules with known labels",
        "      Columns: SMILES, TARGET",
        "   2. unlabeled_pool.csv - Molecules needing prediction", 
        "      Columns: SMILES",
        "   3. oracle_data.csv - Ground truth for all molecules",
        "      Columns: SMILES, TARGET",
        "",
        "✅ **Data Requirements:**",
        "   • SMILES strings must be valid and canonical",
        "   • TARGET values should be numeric (0-1 for classification, any range for regression)",
        "   • No duplicates between initial_labeled and unlabeled_pool",
        "   • Oracle must contain all SMILES from both sets",
        "   • Minimum 3-5 initial labeled molecules recommended",
        "",
        "🎯 **Best Practices:**",
        "   • Include diverse initial molecules (different scaffolds)",
        "   • Balance TARGET distribution in initial set",
        "   • Ensure SMILES are standardized (use RDKit)",
        "   • Remove invalid/problematic SMILES",
        "   • Consider molecular weight and complexity diversity",
        "",
        "⚠️ **Common Issues:**",
        "   • Invalid SMILES strings",
        "   • Missing values in TARGET column",
        "   • Duplicate SMILES across files",
        "   • Non-numeric TARGET values",
        "   • Empty datasets"
    ]
    
    for guideline in guidelines:
        print(guideline)

def validation_example():
    """
    Example of data validation process
    """
    print("\n=== Data Validation Example ===")
    
    # Create example with intentional issues for demonstration
    print("Creating test data with potential issues...")
    
    # Good data
    good_labeled = pd.DataFrame({
        'SMILES': ['CCO', 'CCN'],
        'TARGET': [0.25, 0.75]
    })
    
    good_unlabeled = pd.DataFrame({
        'SMILES': ['CCC', 'CCCO']
    })
    
    good_oracle = pd.DataFrame({
        'SMILES': ['CCO', 'CCN', 'CCC', 'CCCO'],
        'TARGET': [0.25, 0.75, 0.45, 0.65]
    })
    
    print("✅ Testing good data...")
    try:
        validate_input_data(good_labeled, good_unlabeled, good_oracle)
        print("   Good data validation: PASSED")
    except Exception as e:
        print(f"   Unexpected error: {e}")
    
    # Test with duplicate SMILES
    print("\n❌ Testing data with duplicate SMILES...")
    bad_unlabeled = pd.DataFrame({
        'SMILES': ['CCO', 'CCCO']  # CCO is already in labeled
    })
    
    try:
        validate_input_data(good_labeled, bad_unlabeled, good_oracle)
        print("   Should have failed but didn't!")
    except AssertionError as e:
        print(f"   Expected error caught: {e}")
    
    # Test with missing oracle data
    print("\n❌ Testing data with incomplete oracle...")
    incomplete_oracle = pd.DataFrame({
        'SMILES': ['CCO', 'CCN'],  # Missing CCC and CCCO
        'TARGET': [0.25, 0.75]
    })
    
    try:
        validate_input_data(good_labeled, good_unlabeled, incomplete_oracle)
        print("   Should have failed but didn't!")
    except AssertionError as e:
        print(f"   Expected error caught: {e}")

if __name__ == '__main__':
    print("Molecular Active Learning - Input Preparation Guide")
    print("=" * 60)
    
    # Run all examples
    example_1_drug_discovery_scenario()
    example_2_material_science_scenario() 
    example_3_toxicity_prediction_scenario()
    
    # Show guidelines
    data_preparation_guidelines()
    
    # Demonstrate validation
    validation_example()
    
    print(f"\n🎉 Input preparation examples completed!")
    print(f"You now have multiple CSV file sets ready for active learning.")
    print(f"Use any of these examples or follow the guidelines to prepare your own data.") 