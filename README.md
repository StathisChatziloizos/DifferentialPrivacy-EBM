# Differential Privacy for Machine Learning

This repository contains code and experiments based on the paper ["Accuracy, Interpretability, and Differential Privacy via Explainable Boosting"](https://arxiv.org/pdf/2106.09680) by Nori et al. (2021).

## Our implementation includes:

- **benchmark_algorithms.ipynb**  
  Benchmarks non-DP models (Logistic Regression, Random Forest, XGBoost, APLR, and EBM) on several classification datasets (Breast Cancer, Adult, Credit Card Fraud, and Telco Churn). It also computes and prints dataset statistics.

- **custom-DP-EBM.ipynb**  
  A simplified, "toy" implementation of a differentially private Explainable Boosting Machine (DP-EBM). This notebook demonstrates:
  - Data loading and quantile-based binning for numeric features (and direct mapping for categorical features).
  - A cyclic boosting procedure with Gaussian noise injection on residual updates.
  - Model evaluation on the Adult, Telco Churn, and Credit Card Fraud datasets.
- **Presentation.pdf**  
  A short set of slides summarizing our approach, experiments, and findings.

- **Report.pdf**  
  A more detailed write-up of the methodology, results, and our contributions.

## Datasets

All required datasets (Adult, Telco Churn, Credit Card Fraud, and Breast Cancer) are provided or downloaded via URL in the notebooks.  
**Note:** The Heart Disease dataset is **not** included due to size limitations. To use it, please download it manually from [UCI Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

## How to Run

1. Install the required Python packages (e.g., scikit-learn, pandas, numpy, interpret, xgboost).
2. Run the Jupyter notebooks:
   - `benchmark_algorithms.ipynb` for benchmarking non-DP models.
   - `custom-DP-EBM.ipynb` for running the custom DP-EBM experiments.


## Additional Note

Our custom DP-EBM implementation is a simplified version that captures the core idea (cyclic boosting with noise injection) but omits several advanced privacy-preserving steps (e.g., differentially private binning, random split selection, full privacy budget accounting, and bagging) as described in the paper.