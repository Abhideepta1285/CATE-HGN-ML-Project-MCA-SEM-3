# CATE-HGN-ML-Project-MCA-SEM-

# CATE-HGN: Citation Prediction Using Heterogeneous Graph Neural Networks

## Project Overview

This project implements a simplified CATE-HGN-style model for academic paper citation prediction, inspired by the research paper:

**Revisiting Citation Prediction with Cluster-Aware Text-Enhanced Heterogeneous Graph Neural Networks (IEEE ICDE 2023).**

The objective is to explore how textual information and heterogeneous academic relationships can help predict research paper impact.

## Environment

- Python 3.12
- Jupyter Notebook
- PyTorch
- CUDA-enabled NVIDIA GPU (optional)
- NumPy
- Pandas
- Scikit-learn
- SciPy
- Matplotlib

## Dataset

**DBLP V11**

The dataset contains academic research information, including:
- Research papers
- Authors
- Publication venues
- Titles and abstracts
- Citation counts
- Publication years

The project uses a smaller sampled subset to make experiments feasible on systems with limited RAM.

## Memory-Efficient Configuration

The updated Cell 4 uses reservoir sampling to avoid loading the complete dataset into memory.

```python
MAX_PAPERS = 500
MAX_TERMS = 1000
MAX_TERMS_PER_PAPER = 5

MIN_YEAR, MAX_YEAR = 1990, 2020
```

Additional memory-saving techniques:
- Process JSONL records one line at a time.
- Retain only required paper information.
- Limit the number of authors per paper.
- Limit abstract length.
- Keep only a fixed-size sample in memory.

## Project Workflow

1. Dataset loading and inspection.
2. Memory-efficient JSONL parsing.
3. Paper sampling.
4. Text preprocessing and feature extraction.
5. Temporal train-validation-test splitting.
6. Heterogeneous information preparation.
7. Model training.
8. RMSE evaluation.
9. Results analysis.

## Temporal Data Split

- Training: Papers published before 2014.
- Validation: Papers published in 2014.
- Testing: Papers published from 2015 to 2020.

## Running the Project

1. Install the required Python libraries.
2. Download and extract the DBLP V11 dataset.
3. Set the correct dataset path in the notebook.
4. Run the notebook cells sequentially.
5. Monitor system RAM and GPU memory during training.
6. Evaluate the model using RMSE.

## Important Note

This project is an experimental, simplified CATE-HGN-style implementation. It does not yet reproduce every component of the original research paper. Results should not be presented as an exact reproduction of the published CATE-HGN results.

The initial dataset size is intentionally limited to 500 papers to establish a stable execution pipeline before scaling up.
