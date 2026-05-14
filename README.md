# 🏦 End-to-End Anti-Money Laundering (AML) Detection Pipeline

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-orange?style=for-the-badge&logo=xgboost&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**A production-grade, machine learning-driven pipeline for detecting financial money laundering transactions — featuring unsupervised behavioral profiling, XGBoost classification, and incremental continuous learning.**

*Developed by [Abdulrahman Hattem Mohammed](https://github.com/your-username)*

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Pipeline Architecture](#-pipeline-architecture)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Installation](#-installation)
- [Usage](#-usage)
- [Phase Breakdown](#-phase-breakdown)
- [Results & Metrics](#-results--metrics)
- [Key Design Decisions](#-key-design-decisions)
- [Continuous Learning](#-continuous-learning)
- [License](#-license)

---

## 🔍 Overview

Financial institutions process **millions of transactions daily**. Traditional rule-based AML systems flood investigators with false positives, while missing real threats exposes banks to massive regulatory fines. Regulators enforce strict deadlines — e.g., **45 business days** to close every alert.

This project replaces static rule-based systems with a **5-phase ML pipeline** that:

- Learns normal vs. abnormal behavioral patterns from unlabeled transaction data
- Classifies transactions as illicit or legitimate with high recall
- Continuously adapts to new criminal tactics via incremental batch learning — without retraining from scratch

---

## 🏗 Pipeline Architecture

```
Raw Transaction Logs
        │
        ▼
┌─────────────────────┐
│  Phase 1            │  ── Drop leaky columns, encode categoricals,
│  Data Preprocessing │     StandardScaler normalization
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 2            │  ── PCA on all continuous features
│  Dimensionality     │     Retain components explaining 95% variance
│  Reduction (PCA)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 3            │  ── K-Means clustering (k=6, Elbow Method)
│  Behavioral         │     Append `behavioral_segment` as new feature
│  Segmentation       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 4            │  ── XGBoost with scale_pos_weight for imbalance
│  XGBoost Classifier │     Evaluate on Precision, Recall, F1-Score
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Phase 5            │  ── Incremental batch update via xgb_model param
│  Continuous         │     No historical data re-used
│  Learning           │
└─────────────────────┘
         │
         ▼
  Fraud Predictions + Updated Model
```

---

## 📊 Dataset

This project uses the **[PaySim Synthetic Financial Dataset](https://www.kaggle.com/datasets/ealaxi/paysim1)** — a simulation of mobile money transactions based on a real dataset from a financial company, with injected money laundering behavior.

| Feature | Description |
|---|---|
| `step` | Time step (1 step = 1 hour) |
| `type` | Transaction type (CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER) |
| `amount` | Transaction amount |
| `nameOrig` | Customer initiating the transaction |
| `oldbalanceOrg` | Initial balance of origin account |
| `newbalanceOrig` | New balance of origin account |
| `nameDest` | Recipient of the transaction |
| `oldbalanceDest` | Initial balance of destination account |
| `newbalanceDest` | New balance of destination account |
| `isFraud` | ✅ Target label (1 = fraud, 0 = legitimate) |
| `isFlaggedFraud` | Rule-based flag (dropped — data leakage risk) |

> **Download:** Place `paysim.csv` in the project root before running notebooks.

---

## 📁 Project Structure

```
aml-detection-pipeline/
│
├── notebooks/
│   ├── 01_preprocessing.ipynb        # Phase 1: Data cleaning & feature engineering
│   ├── 02_pca.ipynb                  # Phase 2: PCA dimensionality reduction
│   ├── 03_kmeans_clustering.ipynb    # Phase 3: Behavioral segmentation
│   ├── 04_xgboost_classifier.ipynb   # Phase 4: Supervised fraud classification
│   └── 05_continuous_learning.ipynb  # Phase 5: Incremental batch training
│
├── src/
│   ├── preprocessing.py              # Reusable preprocessing functions
│   ├── pca_utils.py                  # PCA helpers & scree plot generation
│   ├── clustering.py                 # K-Means pipeline utilities
│   ├── model.py                      # XGBoost training & evaluation
│   └── continuous_learning.py        # Batch update logic
│
├── outputs/
│   ├── models/                       # Saved XGBoost model files
│   ├── figures/                      # Generated plots (scree, elbow, confusion matrix)
│   └── reports/                      # Classification reports
│
├── paysim.csv                        # Dataset (download from Kaggle — not tracked by git)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-username/aml-detection-pipeline.git
cd aml-detection-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download `paysim.csv` from [Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1) and place it in the project root.

---

## 🚀 Usage

Run the notebooks sequentially, or execute the full pipeline via script:

```bash
# Run full pipeline end-to-end
python src/pipeline.py --data paysim.csv

# Or open notebooks in order
jupyter notebook notebooks/01_preprocessing.ipynb
```

Each notebook is self-contained and outputs artifacts consumed by the next phase.

---

## 🔬 Phase Breakdown

### Phase 1 — Data Preprocessing

- Drops `nameOrig`, `nameDest` (identifiers, not signals) and `isFlaggedFraud` (data leakage)
- Handles null values via `dropna()`
- One-hot encodes `type` column (`drop_first=True` to avoid dummy variable trap)
- Applies `StandardScaler` to all numerical features

```python
df = df.drop(['nameOrig', 'nameDest', 'isFlaggedFraud'], axis=1)
df = pd.get_dummies(df, columns=['type'], drop_first=True)
X_scaled = StandardScaler().fit_transform(X)
```

---

### Phase 2 — PCA Dimensionality Reduction

- Fits PCA across all components
- Selects the minimum number of components explaining **≥95% cumulative variance**
- Produces scree plot and cumulative variance curve

```python
pca = PCA()
cumulative_variance = pca.fit(X_scaled).explained_variance_ratio_.cumsum()
n_components = np.argmax(cumulative_variance >= 0.95) + 1
```

> **Why 95%?** It retains nearly all signal while removing noise and correlated dimensions that would distort K-Means distance calculations.

---

### Phase 3 — K-Means Behavioral Segmentation

- Samples 10,000 rows for fast Elbow Method computation
- Selects `k=6` clusters based on inertia inflection point
- Fits K-Means on the full PCA-reduced dataset
- Appends `behavioral_segment` to the main dataframe as an engineered feature

```python
kmeans = KMeans(n_clusters=6, init='k-means++', random_state=42, n_init=10)
df_final['behavioral_segment'] = kmeans.fit_predict(X_pca_final)
```

> **Why cluster before classifying?** Behavioral context (e.g., "this account type typically bursts CASHOUT transactions") is a stronger fraud signal than any single transaction attribute alone.

---

### Phase 4 — XGBoost Fraud Classifier

- Trains on original scaled features **+** `behavioral_segment`
- Uses `scale_pos_weight=99` to address ~1:99 class imbalance
- Evaluates with Precision, Recall, and F1-Score

```python
model = XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=99,
    eval_metric='logloss'
)
model.fit(X_train, y_train)
```

> ⚠️ **Why not Accuracy?** With ~0.13% fraud rate, a model predicting *all legitimate* achieves 99.87% accuracy — while catching zero fraud cases. Recall and F1-Score expose this failure.

---

### Phase 5 — Continuous Learning

- Simulates a new monthly transaction batch arriving
- Updates the existing model using **only the new batch** — no historical data re-used
- Uses XGBoost's native `xgb_model` parameter for true incremental learning

```python
# Update existing model with new batch only
model.fit(
    X_new, y_new,
    xgb_model=model.get_booster()   # ← Extends existing trees
)
```

> **Strict constraint:** combining old + new data and retraining from scratch is explicitly forbidden — this violates data retention compliance and defeats the purpose of online learning.

---

## 📈 Results & Metrics

| Metric | Score |
|---|---|
| **Recall** | ~93% |
| **Precision** | ~88% |
| **F1-Score** | ~90% |
| Accuracy | ~99.9% *(misleading — see above)* |

> Exact results vary by random seed and train/test split. The focus metric is **Recall** — minimizing missed fraud cases.

---

## 🧠 Key Design Decisions

| Decision | Reason |
|---|---|
| Drop `isFlaggedFraud` | Prevents data leakage from existing rule-based system |
| PCA before K-Means | Denoised space produces more stable, meaningful clusters |
| 95% variance threshold | Retains signal, removes noise, justified by scree plot |
| `scale_pos_weight` over SMOTE | Avoids synthetic sample generation; faster and cleaner |
| Recall-optimized evaluation | A missed fraud is costlier than a false alarm in AML |
| Incremental batch update | Adapts to new patterns; respects data retention regulations |

---

## 🔄 Continuous Learning

The continuous learning module simulates a **real-world production scenario** where new transaction data arrives monthly. Instead of:

❌ Combining all historical + new data → retrain from scratch

The pipeline does:

✅ Load saved model → fit on new batch only → save updated model

This approach is:
- **Computationally efficient** — no full retraining cost
- **Regulatory compliant** — old data is never re-processed
- **Adaptive** — model weights shift toward emerging fraud patterns

---


<div align="center">
  <sub>Built as a final project for an advanced Machine Learning course — replacing the traditional exam with a real-world end-to-end ML system.</sub>
</div>
