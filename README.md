# Molecular Active Learning

A Python library for active learning on molecular datasets using UniMol models and BAAL (Bayesian Active Learning Library).

## Features

- Active learning for molecular property prediction
- Integration with UniMol models
- BALD (Bayesian Active Learning by Disagreement) heuristic
- Support for SMILES-based molecular datasets
- Configurable training parameters and query strategies

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from molecular_active_learning import MolecularActiveLearning
from molecular_active_learning.utils import load_data_from_files

# Load data from separate CSV files (real-world scenario)
initial_labeled, unlabeled_pool, oracle_data = load_data_from_files(
    initial_labeled_path='initial_labeled.csv',   # SMILES + TARGET
    unlabeled_pool_path='unlabeled_pool.csv',     # SMILES only
    oracle_data_path='oracle_data.csv'            # Complete ground truth
)

# Initialize active learning
al = MolecularActiveLearning(
    initial_labeled_data=initial_labeled,
    unlabeled_pool=unlabeled_pool,
    oracle_data=oracle_data,
    batch_size=3,
    max_iterations=5
)

# Run active learning
results = al.run_active_learning()
```

## Data Format

Your CSV files should follow this format:

### Labeled Data (initial_labeled.csv)
```csv
SMILES,TARGET
CCO,0.789
CCN,0.456
...
```

### Unlabeled Pool (unlabeled_pool.csv)
```csv
SMILES
CCC
CCN(C)C
...
```

## Examples

See the `examples/` directory for detailed usage examples:

- `basic_example.py`: Simple workflow demonstration
- `advanced_example.py`: Custom configurations and strategy comparison
- `visualization_example.py`: Result analysis and visualization

## Project Structure

```
molecular-active-learning/
├── README.md                  # Project overview and quick start
├── LICENSE                    # MIT license
├── requirements.txt           # Core dependencies
├── requirements-dev.txt       # Development dependencies
├── setup.py                   # Package installation script
├── .gitignore                 # Git ignore patterns
├── CONTRIBUTING.md            # Contribution guidelines
├── src/
│   └── molecular_active_learning/
│       ├── __init__.py        # Package initialization
│       ├── core/              # Core functionality
│       │   ├── dataset.py     # Dataset classes
│       │   ├── model_wrapper.py # Model interface
│       │   └── active_learner.py # Main AL logic
│       └── utils/             # Utility functions
│           └── helpers.py     # Helper functions
├── examples/                  # Usage examples
│   ├── basic_example.py
│   ├── advanced_example.py
│   ├── visualization_example.py
│   └── data/
│       └── sample_data.csv
├── tests/                     # Test suite
│   ├── test_basic.py
│   └── test_advanced.py
└── docs/                      # Documentation
    └── usage.md
```

## Dependencies

- Python (3.7+)
- PyTorch
- unimol-tools
- baal
- pandas
- numpy
- rdkit
- scikit-learn
``` 