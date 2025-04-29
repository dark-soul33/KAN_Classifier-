# KAN for Malicious PDF Detection

This project benchmarks the performance of **Kolmogorov-Arnold Networks (KAN)** on detecting malicious PDF files using metadata-based structural features. It includes model training, evaluation using various classification metrics, and interpretation through LIME.

## 📘 Overview

**Kolmogorov-Arnold Networks (KAN)** use spline-based non-linearities and mathematical interpolation, diverging from traditional deep neural networks. This study applies KAN to a cybersecurity problem: predicting if a PDF file is benign or malicious based on static feature analysis.

- **Original KAN paper**: [arXiv:2404.19756](https://arxiv.org/abs/2404.19756)
- **KAN source repo**: [efficient-kan](https://github.com/Blealtan/efficient-kan)

## 📁 Project Files

```
.
├── requirements.txt       # modules required 
├── kan_modules.py         # Core KAN implementation
├── kan_sklearn.py         # Scikit-learn wrappers for KAN/KANGAM
├── KAN_dataset1.ipynb     # Experiment notebook with detailed metrics and LIME
├── KAN_dataset2.ipynb     # Additional dataset experiment (optional)
├── pdfdataset_cleaned.csv  # Preprocessed dataset (from Contagio 2022)
├── pdfFeatureFile_cleaned.csv  # another dataset
└── README.md
```

## 🧪 Dataset

The dataset is derived from **Contagio PDF Malware Dataset (2022)** and preprocessed into `pdfdataset_cleaned.csv`. Features include structural properties like object counts, JavaScript usage, and embedded stream patterns — commonly used in static PDF malware detection.


## ⚙️ Preprocessing

- Feature scaling using `StandardScaler`
- Stratified train-test split (80:20)
- Model trained on scaled inputs

## 🧠 Model

This project uses:

- `KANClassifier`: Dense-layer KAN
- Hyperparameter grid search across:
  - Hidden layer sizes: `[32, 64]`
  - Activation regularization: `[0.1, 0.3]`
  - Entropy regularization: `[0.1, 0.3]`
  - Ridge regularization: `[0.1, 0.3]` (for `KANGAM`)

The final model uses:
```python
KANClassifier(hidden_layer_size=32, device='cpu',
              regularize_activation=0.3,
              regularize_entropy=0.3,
              regularize_ridge=0.1,
              spline_order=3)
```

## 📊 Metrics and Evaluation

Metrics computed include:

- **Accuracy** on test set
- **Classification report**: precision, recall, F1
- **Cohen’s Kappa Score** for agreement analysis
- **Confusion Matrix** (raw + heatmap)
- **ROC Curve + AUC**
- **Precision-Recall Curve + Average Precision**
- **Sensitivity / Specificity**
- **Matthews Correlation Coefficient (MCC)**

## 🧠 Model Interpretability with LIME

- Local Interpretable Model-agnostic Explanations (LIME) are used to analyze individual predictions.
- Explains how each input feature contributes to a classification decision.
- Output includes visual plots and saved HTML explanations.

## 📦 Installation

```bash
pip install -r requirements.txt
```

Requirements:
- `torch`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `lime`
- `tqdm`
- `numpy`
- `pandas`

## 🚀 Running the Notebook

Launch the experiment with:

```bash
jupyter notebook KAN_dataset1.ipynb
```

## 📝 Citation & Credits

This work builds on:

- [efficient-kan](https://github.com/Blealtan/efficient-kan)
- [pykan](https://github.com/KindXiaoming/pykan)
- [KAN Paper](https://arxiv.org/abs/2404.19756)

If using this project, please cite the original paper and repos accordingly.

## 📬 Contact

Questions or suggestions? Please open an issue or contribute via pull request.
