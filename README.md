# ECGNextFormer
### A Transformer-Based Framework for ECG Analysis with Rare-Shot Learning and Age-Aware Evaluation

<p align="center">
  <b>ECGNextFormer: Research Implementation</b>
</p>

---

## Overview

**ECGNextFormer** is a research project focused on deep learning-based electrocardiogram (ECG) analysis, with particular emphasis on robust learning under limited data availability and evaluation across different patient age groups.

The project investigates Transformer-based ECG representation learning alongside experimental strategies for improving performance in rare or underrepresented data scenarios. It explores sampling and augmentation strategies, alternative optimization objectives, and age-aware evaluation to better understand model behavior across different population subgroups.

The implementation includes model training pipelines, dataset analysis utilities, experimental notebooks, and evaluation workflows.

## Research Objectives

- Develop and investigate a Transformer-based architecture for ECG signal analysis.
- Study the impact of sampling strategies on model performance.
- Evaluate augmentation techniques under rare-shot learning settings.
- Investigate alternative loss functions and their influence on model performance.
- Analyze performance across different age groups.
- Examine error patterns and limitations using quantitative and visual evaluation.

## Methodology

The research workflow consists of the following stages:

1. **Dataset Preparation:** Process ECG recordings and associated metadata for model development and evaluation.
2. **Signal Analysis:** Perform dataset inspection, metadata auditing, and age-related distribution analysis.
3. **Model Development:** Implement the ECGNextFormer architecture and associated training pipelines.
4. **Experimental Evaluation:** Compare baseline configurations against sampling and augmentation-based variants.
5. **Age-Aware Analysis:** Investigate performance differences across age groups.
6. **Error Analysis:** Examine prediction errors, confusion patterns, and relevant evaluation metrics.

## Model Architecture

ECGNextFormer is implemented using a Transformer-based architecture for ECG signal modeling.

The model implementation and training pipelines are available in the `code/` directory.

| Component | Description |
|---|---|
| ECGNextFormer | Transformer-based ECG model implementation |
| Training Pipeline | Model training and experiment execution |
| Pooled Analysis | Analysis utilities for pooled evaluation |
| Evaluation Notebooks | Experimental evaluation and visualization |

## Dataset

This project uses the **PTB-XL ECG Dataset**, a publicly available electrocardiography dataset.

The original ECG recordings are not included in this repository.

Dataset information and access:

- [PTB-XL Dataset — PhysioNet](https://physionet.org/content/ptb-xl/1.0.3/)

Please refer to the dataset's official documentation for access conditions, licensing, and usage requirements.

### Dataset Preparation

The dataset must be downloaded and prepared separately before executing the training or evaluation pipelines.

The repository includes selected dataset analysis reports and metadata-related outputs.

## Repository Structure

```text
DS_RESEARCH/
│
├── code/
│   ├── analyze_pooled.py
│   ├── ecgnextformer.py
│   ├── train_ecgnextformer.py
│   └── train_ecgnextformer_v2.py
│
├── dataset_reports/
│   ├── age_distribution.png
│   ├── exact_age_frequency_counts.csv
│   ├── excluded_records.csv
│   ├── metadata_age_valid.csv
│   ├── metadata_audit_summary.csv
│   └── missing_waveform_files.csv
│
├── ECGNextFormer_Evaluation_Figures.ipynb
├── ECGNextFormer_RareShot_Complete.ipynb
├── requirements.txt
├── README.md
└── .gitignore
```

**Note:** Experimental results and selected evaluation artifacts will be organized under `results/` as they are prepared for publication.

## Installation

### Prerequisites

- Python 3.x
- pip
- Git
- Compatible CPU or GPU environment
- NVIDIA GPU and CUDA, if GPU acceleration is required

### Clone the Repository

```bash
git clone https://github.com/Bhaskar9312/DS_RESEARCH.git
cd DS_RESEARCH
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

For reproducible experiments, use a dedicated Python virtual environment and compatible library versions.

## Usage

The repository contains training scripts, model implementations, and evaluation notebooks.

### Model Implementation

```text
code/ecgnextformer.py
```

### Training Pipelines

```text
code/train_ecgnextformer.py
code/train_ecgnextformer_v2.py
```

### Pooled Analysis

```text
code/analyze_pooled.py
```

Example execution:

```bash
python code/analyze_pooled.py
```

Training and evaluation commands may require dataset paths, configuration parameters, and experiment-specific arguments. Refer to the respective scripts and notebooks for their complete execution requirements.

## Experimental Investigations

The research explores multiple experimental configurations, including:

- Baseline model configurations
- Sampling-based approaches
- Data augmentation strategies
- Sampling combined with augmentation
- Alternative loss-function configurations
- Rare-shot evaluation
- Age-group-based performance analysis

The objective is to investigate how these approaches influence predictive performance and generalization under different data conditions.

## Evaluation and Analysis

The evaluation workflow includes:

- Quantitative performance comparisons
- Confusion matrix analysis
- Age-wise evaluation
- Error distribution analysis
- Training behavior visualization
- Comparative experimental analysis

Detailed evaluation procedures and visualizations are available in the research notebooks.

## Dataset Analysis

The `dataset_reports/` directory contains supporting dataset analysis artifacts, including:

- Age distribution visualization
- Exact age frequency statistics
- Excluded record information
- Metadata age validation
- Metadata audit summary
- Missing waveform file analysis

These artifacts support dataset inspection and experimental transparency.

## Reproducibility

For reproducible experiments:

1. Use the same dataset version and preprocessing configuration.
2. Install compatible dependency versions.
3. Preserve experiment configurations and random seeds.
4. Use consistent train, validation, and test splits.
5. Record model configurations and evaluation metrics.
6. Maintain experiment-specific logs and outputs.
7. Document hardware and software environments when reporting results.

## Limitations

This repository represents an ongoing research implementation.

Experimental conclusions should be interpreted in the context of the dataset, evaluation protocol, class distribution, and experimental configurations used.

Performance comparisons should be supported by consistent evaluation procedures and appropriate statistical analysis.

## Future Work

Potential directions include:

- Further investigation of rare-shot learning strategies.
- Improved robustness across age groups.
- More comprehensive evaluation across ECG subpopulations.
- Additional ablation studies.
- Improved model efficiency and reproducibility.
- Consolidation of final experimental results and research figures.

## Authors

Research team — ECGNextFormer

## Acknowledgments

We acknowledge the authors and maintainers of the PTB-XL dataset for making publicly available ECG data accessible for research.

## Citation

If you use this implementation in academic work, please cite the associated research publication when available.

## License

A license has not yet been specified for this repository. Please refer to the repository owner for permissions regarding reuse, redistribution, and modification.
