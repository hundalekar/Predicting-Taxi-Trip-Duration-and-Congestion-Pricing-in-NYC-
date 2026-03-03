# 🚕 Predicting Taxi Trip Duration & Congestion Pricing Exposure in New York City

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/ML-Scikit--Learn%20%7C%20XGBoost-green.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-brightgreen.svg)]()
[![Course](https://img.shields.io/badge/Course-IST%20707%20Applied%20Machine%20Learning-purple.svg)](https://ischool.syr.edu/)

> **A dual-model machine learning system for NYC taxi drivers to optimize trip decisions in the post-congestion pricing era.**

**Abhishek Hundalekar**

---

## 📋 Table of Contents

- [Project Overview](#-project-overview)
- [What Makes This Project Novel](#-what-makes-this-project-novel)
- [Dataset](#-dataset)
- [Project Structure](#-project-structure)
- [Methodology](#-methodology)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Model Validation](#-model-validation)
- [Technologies Used](#-technologies-used)
- [Setup & Installation](#-setup--installation)
- [How to Run](#-how-to-run)
- [Author](#-author)

---

## 🎯 Project Overview

New York City's congestion pricing program launched on **January 5, 2025**, adding a $9 surcharge for vehicles entering Manhattan's Central Business District (CBD) below 60th Street. This fundamentally changed the economics of taxi driving in NYC.

I built **two predictive models** to help taxi drivers make better trip acceptance decisions:

| Model | Task | Question It Answers |
|-------|------|-------------------|
| **Model 1** | Regression | *"How long will this trip take?"* — Predicts trip duration in minutes |
| **Model 2** | Classification | *"Will the congestion fee apply?"* — Predicts congestion pricing exposure (Yes/No) |

### Primary Stakeholder

**Maria Santos** — a composite persona representing NYC taxi drivers and fleet operators. Maria needs to quickly evaluate ride requests: estimate trip duration to calculate her effective hourly rate, and know whether the congestion fee applies to understand total passenger cost and demand patterns.

### Success Metrics

| Model | Metric | Target | Achieved |
|-------|--------|--------|----------|
| Model 1 (Duration) | RMSE | < 10 minutes | **3.50 minutes** ✅ (65% below target) |
| Model 2 (Congestion) | Accuracy | > 85% | **96.7%** ✅ (exceeds by 11.7pp) |

---

## 🆕 What Makes This Project Novel

### 1. Post-Congestion Pricing Data
All existing ML work on NYC taxi data uses **pre-2025 data** without congestion pricing:
- [Kaggle NYC Taxi Trip Duration](https://www.kaggle.com/c/nyc-taxi-trip-duration) — uses 2016 data
- [Top GitHub solutions](https://github.com/pklauke/Kaggle-NYCTaxi) — pre-congestion pricing era

**My contribution:** I use **January 2025 data** with actual `cbd_congestion_fee` values — one of the first ML analyses of real taxi trips after congestion pricing was implemented.

### 2. Dual-Model Operational Support
Existing tools predict duration **OR** cost in isolation. This project combines both into an integrated decision-support system, enabling drivers to simultaneously estimate trip time and fee exposure.

---

## 📊 Dataset

| Property | Detail |
|----------|--------|
| **Source** | [NYC Taxi & Limousine Commission (TLC)](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) |
| **Period** | January 2025 |
| **Raw Size** | 3,472,210 trip records |
| **After Cleaning** | 2,721,501 (Model 1) / 2,412,426 (Model 2) |
| **Features Engineered** | 62 (Model 1) / 59 (Model 2) |
| **Format** | Parquet |

### Feature Categories
- **Temporal (11):** Pickup hour, day of week, weekend/rush hour flags, cyclical encodings
- **Geographic (11):** Pickup/dropoff zone IDs, CBD flags, airport indicators, zone popularity
- **Trip Characteristics (10):** Distance, distance bins, passenger count, rate codes
- **Encoding & Interactions (~16):** One-hot encoded categoricals, distance × passenger, hour × day of week
- **CBD Trip Types:** Within CBD, into CBD, out of CBD, outside CBD

> **Note:** The raw parquet file (~59 MB) is not included in the repository due to size. See [Setup & Installation](#-setup--installation) for download instructions.

---

## 📁 Project Structure

```
ML-Project/
│
├── README.md                          ← You are here
├── requirements.txt                   ← Python dependencies
├── LICENSE                            ← MIT License
├── .gitignore                         ← Excludes large data/model files
│
├── data/
│   ├── raw/                           ← Original TLC data (not in repo)
│   │   └── yellow_tripdata_2025-01.parquet
│   ├── processed/                     ← Cleaned & engineered datasets
│   │   ├── cleaned_model1_duration.parquet
│   │   ├── cleaned_model2_congestion.parquet
│   │   ├── fe_model1_duration.parquet
│   │   ├── fe_model2_congestion.parquet
│   │   ├── analysis_model1.parquet
│   │   ├── analysis_model2.parquet
│   │   ├── baseline_results_model1.csv
│   │   └── baseline_results_model2.csv
│   └── README.md                      ← Data documentation
│
├── notebooks/
│   ├── 01_data_exploration.ipynb      ← EDA with 9+ visualizations
│   ├── 02_data_cleaning.ipynb         ← Outlier removal, missing values
│   ├── 03_feature_engineering.ipynb   ← 62/59 features created
│   ├── 04_baseline_models.ipynb       ← Linear & Logistic Regression
│   └── 05_advanced_models.ipynb       ← Random Forest & XGBoost
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py            ← Data loading & cleaning functions
│   ├── feature_engineering.py         ← Feature creation pipeline
│   ├── models.py                      ← Model training & tuning
│   └── evaluation.py                  ← Metrics & diagnostic plots
│
├── results/
│   ├── figures/                       ← Diagnostic plots & visualizations
│   └── model_performance/             ← Metrics CSVs & comparisons
│
└── docs/
    └── project_report.md              ← Full project documentation
```

---

## 🔬 Methodology

### Pipeline Overview

```
Raw Data (3.47M trips)
    ↓
[1] Exploratory Data Analysis — Distributions, patterns, quality issues
    ↓
[2] Data Cleaning — Remove invalid trips, handle missing values, create two model-specific datasets
    ↓
[3] Feature Engineering — 62/59 features: temporal, geographic, trip, interaction
    ↓
[4] Baseline Models — Linear Regression (RMSE 4.89) / Logistic Regression (93.6% Acc)
    ↓
[5] Advanced Models — Random Forest & XGBoost with hyperparameter tuning
    ↓
[6] Validation — Leakage audit, overfitting check, feature importance analysis
```

### Data Cleaning Highlights
- Removed **540,149 null rows** (traced to a single vendor's systematic data issue)
- Filtered trips to **1–180 minutes** duration (removed 0-second and 24+ hour anomalies)
- Removed impossible values (0 distance, negative fares, dropoff before pickup)
- Created **two separate cleaned datasets** — Model 1 uses all January 2025 trips; Model 2 excludes pre-Jan-5 trips (congestion pricing not yet active)
- Applied the **"Maria Test"** for data leakage: if Maria doesn't know a value at pickup time, it cannot be a model feature

### Feature Engineering Highlights
- **Cyclical encoding** for hour/day (so 11 PM and midnight are neighbors, not 23 units apart)
- **CBD zone identification** using a data-driven approach (50%+ fee rate threshold → 40 CBD zones identified)
- **Log transform** on trip distance (reduced skewness from 3.22 → 1.32)
- **Analysis-only features** (speed, fare/mile, tip%, cost/minute) separated before modeling to prevent leakage
- **Zero data leakage** verified through automated column scanning

### Algorithms Used

| Algorithm | Type | Why I Used It |
|-----------|------|---------------|
| **Linear Regression** | Baseline | Simplest model — establishes performance floor for duration prediction |
| **Logistic Regression** | Baseline | Classification baseline — linear decision boundary for fee prediction |
| **Random Forest** | Advanced | Ensemble of decision trees — captures non-linear patterns, handles multicollinearity |
| **XGBoost** | Advanced | Gradient boosted trees — each tree corrects previous mistakes. Typically best on tabular data |

### Training Strategy

- **Train-Test Split:** 80/20 with stratification for classification
- **Hyperparameter Tuning:** RandomizedSearchCV (30 iterations × 3-fold CV)
- **Subsample Strategy:** 15% of training data for tuning search, then retrain best parameters on full data
- Tree-based models used **unscaled data** (scaling unnecessary for tree splits)

---

## 📈 Results

### Model 1 — Trip Duration Regression

| Model | MAE (min) | RMSE (min) | R² | Train Time |
|-------|-----------|-----------|-----|------------|
| Predict Mean (naive) | 7.59 | 10.72 | 0.000 | — |
| Linear Regression | 3.23 | 4.89 | 0.792 | 7s |
| Random Forest (tuned) | 2.41 | 3.73 | 0.879 | 1011s |
| **XGBoost (tuned)** 🏆 | **2.25** | **3.50** | **0.893** | **145s** |

**Best Model:** XGBoost — **28.4% RMSE improvement** over Linear Regression baseline, **65% below stakeholder target**

### Model 2 — Congestion Fee Classification

| Model | Accuracy | F1 | AUC-ROC | No Fee Recall | Train Time |
|-------|----------|-----|---------|---------------|------------|
| Predict Majority (naive) | 74.6% | 0.854 | 0.500 | 0.0% | — |
| Logistic Regression | 93.6% | 0.958 | 0.964 | 82.2% | — |
| Random Forest (tuned) | 96.4% | 0.976 | 0.991 | 97.2% | 275s |
| **XGBoost (tuned)** 🏆 | **96.7%** | **0.977** | **0.991** | **95.6%** | **51s** |

**Best Model:** XGBoost — **AUC-ROC 0.991**, minority class recall improved from 82.2% → 95.6%

### Stakeholder Impact

For Maria (taxi driver), the models provide:
- Trip duration predictions accurate within **~2.25 minutes** on average
- Congestion fee predictions correct **96.7%** of the time
- Fee trip detection with **97.0% recall** (misses only 3.0%)
- Both models fast enough for **real-time trip decisions**

---

## 💡 Key Findings

1. **XGBoost outperformed all models** for both tasks — best accuracy AND 7× faster training than Random Forest
2. **Algorithm choice matters more than tuning** — switching from Linear to tree-based models gave 23% improvement; tuning added another 7%
3. **Distance drives duration prediction** — top features are trip_distance, dist_very_long, log_trip_distance
4. **Location drives congestion fee prediction** — 8 of 10 top features are location-based, confirming the fee is fundamentally a geographic policy
5. **Minority class (No Fee) recall dramatically improved** — from 82.2% (Logistic Regression) to 95.6% (XGBoost)
6. **No multicollinearity issues** with tree-based models — Linear Regression's trillion-scale coefficients from one-hot encoding are resolved

---

## ✅ Model Validation

Comprehensive validation was performed to ensure model integrity:

| Check | What Was Verified | Result |
|-------|-----------------|--------|
| **Feature Source Audit** | No post-trip columns (fare, surcharge, tip) in Model 2 features | ✅ Clean |
| **Target Proxy Check** | No feature has \|correlation\| > 0.90 with target | ✅ No proxies |
| **Wait-and-See Test** | All Model 2 features are known or estimable at pickup time | ✅ All available |
| **Geographic Validation** | 6 CBD features alone achieve ~95% accuracy with a 3-level tree | ✅ Fee is a geographic rule |
| **Overfitting Check** | Training vs Test accuracy gap is small for both models | ✅ Models generalize |

### Why Is Model 2 Accuracy So High (96.7%)?

The NYC congestion fee is fundamentally a **rule-based geographic policy**: if the trip touches Manhattan's CBD below 60th Street, the fee applies. The features include direct indicators of this (`pu_is_cbd`, `do_is_cbd`, `cbd_within_cbd`). High accuracy is **expected** — a simple 3-level decision tree using only 6 CBD features achieves ~95% accuracy. The remaining improvement comes from handling edge cases (boundary zones, airport exemptions, transition hours).

This is **not** data leakage — it's the model correctly learning the geography of New York City.

---

## 🛠 Technologies Used

| Category | Tools |
|----------|-------|
| **Language** | Python 3.12 |
| **Data Processing** | Pandas, NumPy |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Visualization** | Matplotlib, Seaborn |
| **Data Storage** | Parquet (Apache Arrow) |
| **Model Persistence** | Joblib |
| **Development** | VS Code, Jupyter Notebooks |
| **Version Control** | Git, GitHub |

---

## ⚙ Setup & Installation

### Prerequisites
- Python 3.10+ installed
- Git installed

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/hundalekar/Predicting-Taxi-Trip-Duration-and-Congestion-Pricing-in-NYC-.git
cd Predicting-Taxi-Trip-Duration-and-Congestion-Pricing-in-NYC-

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the dataset
# Download from: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
# File: Yellow Taxi Trip Records — January 2025
# Place in: data/raw/yellow_tripdata_2025-01.parquet
```

### requirements.txt

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
matplotlib>=3.7
seaborn>=0.12
joblib>=1.3
pyarrow>=12.0
```

---

## 🚀 How to Run

Run the notebooks in order — each builds on the previous:

```
1. notebooks/01_data_exploration.ipynb    → Understand the data
2. notebooks/02_data_cleaning.ipynb       → Clean & prepare
3. notebooks/03_feature_engineering.ipynb  → Create 62/59 features
4. notebooks/04_baseline_models.ipynb     → Linear & Logistic Regression
5. notebooks/05_advanced_models.ipynb     → Random Forest & XGBoost (best results)
```

> **Note:** The advanced models notebook requires ~2–3 hours of compute time for full training and hyperparameter tuning on a standard laptop.

---

## 👤 Author

**Abhishek Hundalekar**
Master's Student, School of Information Studies
Syracuse University

**Course:** IST 707 — Applied Machine Learning | Spring 2026

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **NYC Taxi & Limousine Commission** for providing open trip record data
- **IST 707 Course Staff** at Syracuse University
- NYC's congestion pricing program (launched January 5, 2025) for creating a novel research opportunity

---

<p align="center">
  <strong>Built by Abhishek Hundalekar | Syracuse University</strong>
</p>