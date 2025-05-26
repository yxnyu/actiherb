# Usage Guide

## Basic Usage

### 1. Data Preparation

Your data should be in CSV format with the following structure:

**Complete Dataset:**
```csv
SMILES,TARGET
CCO,0.789
CCN,0.456
CCC,0.234
...
```

### 2. Split Data

You can either manually split your data or use the utility function:

```python
from molecular_active_learning.utils import split_data_randomly
import pandas as pd

# Load complete dataset
complete_data = pd.read_csv('your_data.csv')

# Split randomly
initial_labeled, unlabeled_pool = split_data_randomly(
    complete_data, 
    initial_labeled_fraction=0.1,
    random_state=42
)
```

### 3. Run Active Learning

```python
from molecular_active_learning import MolecularActiveLearning

# Initialize
mal = MolecularActiveLearning(
    task='regression',
    epochs=10,
    batch_size=16,
    query_size=5
)

# Run
final_dataset = mal.run(
    initial_labeled=initial_labeled,
    unlabeled_pool=unlabeled_pool,
    oracle_data=complete_data,
    iterations=20
)
```

## Advanced Configuration

### Custom Parameters

```python
mal = MolecularActiveLearning(
    task='regression',          # or 'classification'
    epochs=20,                  # training epochs per iteration
    batch_size=32,              # training batch size
    query_size=10,              # samples to query per iteration
    metrics='pearsonr',         # evaluation metric
    model_name='unimolv2',      # UniMol model variant
    model_size='1.1B',          # model size
    log_file='custom_log.log'   # custom log file
)
```

### Output Management

```python
final_dataset = mal.run(
    initial_labeled=initial_labeled,
    unlabeled_pool=unlabeled_pool,
    oracle_data=complete_data,
    iterations=20,
    save_intermediate=True,     # save intermediate results
    output_dir='./results/'     # custom output directory
)
```

## Understanding the Output

The system generates several files during execution:

- `train_round_N.csv`: Training data for round N
- `labeled_round_N.csv`: All labeled data after round N
- `unlabeled_round_N.csv`: Remaining unlabeled data after round N
- `model_round_N/`: Trained model directory for round N
- `final_labeled.csv`: Final dataset with all labels
- `active_learning.log`: Detailed execution log

## Tips for Better Results

1. **Start Small**: Begin with a small initial labeled set (5-10% of total data)
2. **Adjust Query Size**: Smaller query sizes often lead to better sample selection
3. **Monitor Logs**: Check the log file for uncertainty scores and sample selection details
4. **Validate Results**: Use the intermediate files to analyze the learning progress 