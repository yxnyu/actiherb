# Contributing to Molecular Active Learning

We welcome contributions to the Molecular Active Learning project! This document provides guidelines for contributing.

## Development Setup

1. Fork and clone the repository:
```bash
git clone https://github.com/yourusername/molecular-active-learning.git
cd molecular-active-learning
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -e .
```

## Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking

Run these tools before submitting:
```bash
black src/ tests/ examples/
flake8 src/ tests/ examples/
mypy src/
```

## Testing

Run tests using pytest:
```bash
pytest tests/ -v --cov=src/molecular_active_learning
```

## Submitting Changes

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes and ensure all tests pass
3. Add tests for new functionality
4. Update documentation if necessary
5. Submit a pull request with a clear description of changes

## Reporting Issues

Please use the GitHub issue tracker to report bugs or request features. Include:
- A clear description of the problem
- Steps to reproduce (for bugs)
- Your environment details (Python version, OS, etc.)
- Example code or data (if applicable)

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to create a welcoming environment for all contributors. 