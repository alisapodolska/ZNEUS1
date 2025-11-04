# ZNEUS Project 1 – COVID-19 Classification  
**Week 1: Data Analysis & Preprocessing**

**Team:** Alisa Podolska, Yulian Kysil  
**Dataset:** https://www.kaggle.com/datasets/meirnizri/covid19-dataset  
**Goal:** Detect COVID-19 cases using patient medical data (initially multiclass → converted to binary classification)

---

##  1. Dataset Overview

| Property | Description |
|----------|-------------|
| Raw dataset size | 1,048,576 rows, 21 columns |
| After removing duplicates | 236,526 rows |
| Final cleaned data | 229,339 rows, 17 features |
| Target column | `CLASSIFICATION_FINAL` |
| Task type | Binary classification |

---

##  2. Data Preprocessing Workflow

###  Loading & Column Fix
- Loaded dataset using `pandas.read_csv()`
- Renamed column `CLASIFFICATION_FINAL → CLASSIFICATION_FINAL`

###  Duplicate Removal
- Detected **812,049 duplicated rows**
- Removed duplicates → kept **236,526 unique records**

###  Missing Values
Values 97 and 99 were treated as missing (NaN).  
Missing value percentage example:

| Column     | Missing % |
|------------|------------|
| PREGNANT   | 54.75%     |
| INTUBED    | 44.56%     |
| ICU        | 44.61%     |

**Actions:**
 Removed columns with >40% missing: `PREGNANT`, `INTUBED`, `ICU`  
 For `PNEUMONIA` missing values → replaced with **2 (No Pneumonia)**

###  Removed Data Leakage
- Dropped column **`DATE_DIED`**, because it contains information after diagnosis.

---

##  3. Feature Engineering

### Target Encoding (Multiclass → Binary)
| Original values | Meaning        | Converted |
|-----------------|----------------|-----------|
| 1–3             | COVID Positive | 1         |
| 4–7             | COVID Negative | 0         |

###  Medical Binary Columns Converted (1/2 → 1/0)
Converted to binary:
`PNEUMONIA, DIABETES, COPD, ASTHMA, INMSUPR, HIPERTENSION, OTHER_DISEASE, CARDIOVASCULAR, OBESITY, RENAL_CHRONIC, TOBACCO`

###  One-Hot Encoding (Categorical → Numeric)
Applied to columns:
- `SEX`, `PATIENT_TYPE`, `MEDICAL_UNIT`

Renamed for clarity:
- `SEX_2 → IS_MALE`  
- `PATIENT_TYPE_2 → IS_HOSPITALIZED`

###  Scaling
- Used **Min-Max Scaler** for `AGE` column.

---

##  4. Train/Validation/Test Split

| Dataset | Percentage | Rows    |
|---------|------------|---------|
| Train   | 70%        | 160,537 |
| Validation | 15%     | 34,401  |
| Test    | 15%        | 34,401  |

✔ Split done using `train_test_split` with stratification.

---

##  5. Correlation & Hypothesis Testing

###  Most correlated features with target (`CLASSIFICATION_FINAL`)
- `PNEUMONIA`
- `AGE`
- `IS_HOSPITALIZED`
- `OTHER_DISEASE`

###  Hypothesis Testing Example (Chi-Square Test)

**H0:** Age is independent of COVID result  
**H1:** Age is related to COVID result

Chi2 = 8656.253, p-value < 0.00001 → Reject H0
There IS a significant relationship between AGE and CLASSIFICATION_FINAL.
---

## 6. Final Output

Saved final processed data to CSV:
X_train.csv, X_val.csv, X_test.csv
y_train.csv, y_val.csv, y_test.csv


**Final dataset properties:**
✅ Fully numerical  
✅ No missing values  
✅ Encoded & normalized  
✅ Ready for MLP model training

---

##   Week 1 Completed  
 Data cleaned, structured, encoded and ready for modeling in Week 2.

---



---

#  Week 2 – Model Development & Baseline MLP

**Objective:** Implement a baseline MLP model using the cleaned dataset from Week 1, build a reusable training pipeline, test multiple architectural improvements (Dropout, BatchNorm, Bottleneck, Skip Connections) and evaluate performance.

---

##  1. Configuration & Baseline Setup

- Model defined in `config.json` for reproducibility.
- Uses **ReLU** activations, **sigmoid** output, hidden layers = `[128, 64, 32]`.
- Optimizer = **Adam**, with early stopping and learning rate `0.001`.

| Section | Parameters |
|---------|------------|
| Model | input_dim = 27, hidden = [128, 64, 32], dropout = 0.3, batch_norm = True |
| Training | batch_size = 64, epochs = 30, optimizer = Adam, patience = 5 |
| Experiment | name = baseline_mlp_dropout_bn, description = Baseline with dropout and BN |

All configurations were saved to `experiments_log.json`.

---

##  2. Data Loading & Preparation

- Loaded processed files from Week 1 (`X_train.csv`, `X_val.csv`, `X_test.csv`, `y_*`).
- Class distribution: COVID Positive ≈ 49.4%, Negative ≈ 50.6% → balanced, no class weights required.
- Final input dimension for the model: **27 features**.

---

##  3. Model Architectures

| Model | Description |
|--------|-------------|
| **build_mlp_a()** | Deep MLP: 512→256→128→64, LeakyReLU, BatchNorm, Dropout(0.1) |
| **build_mlp_b(cfg)** | Fully configurable architecture: custom hidden layers, BatchNorm, Dropout, optional Bottleneck layers & Skip Connections |

- Output layer: `Dense(1, activation='sigmoid')` for binary classification.

---

##  4. Training Pipeline (Reusable Functions)

| Function | Purpose |
|----------|---------|
| `build_and_compile_model(config)` | Creates and compiles MLP model with Adam & metrics (Accuracy, AUC, Precision, Recall) |
| `train_model()` | Trains model with validation split and EarlyStopping |
| `evaluate_model()` | Computes Loss, Acc, AUC, Precision, Recall, F1, and Confusion Matrix on test set |
| `visualize_results()` | Plots learning curves & ROC curve |
| `log_experiment()` | Saves config + metrics to `experiments_log.json` |
| `run_experiment()` | End-to-end pipeline: build → train → evaluate → log |

---

##  5. Experiment Series 1 – Incremental Improvements

| Experiment | Configuration | Accuracy | F1 Score | AUC |
|------------|---------------|----------|----------|-----|
| Baseline | No Dropout, No BN | 0.633 | 0.627 | 0.673 |
| + Dropout | Dropout = 0.3 | 0.633 | 0.616 | 0.673 |
| + BatchNorm | Dropout + BN | 0.631 | 0.632 | 0.672 |
| + Skip Connections | Dropout + BN + Skip | 0.632 | 0.623 | 0.671 |
| + Bottleneck | All techniques | 0.633 | 0.628 | 0.672 |

**Conclusion:** Dropout + BatchNorm gives the most balanced and stable validation results.

---

##  6. Experiment Series 2 – Optimizer Comparison

| Model | Optimizer | LR | Accuracy | F1 | AUC |
|--------|-----------|----|----------|----|-----|
| model_a | AdamW | 1e-4 | 0.625 | 0.638 | 0.660 |
| model_b | AdamW | 1e-4 | 0.626 | 0.623 | 0.672 |
| model_b | SGD + Momentum | 5e-3 | 0.626 | 0.617 | 0.656 |
| model_b | Adam (no BN) | 1e-3 | 0.627 | 0.613 | 0.673 |

 **AdamW** shows stable training, but default **Adam** achieves highest AUC.

---

##  7. Experiment Series 3 – Random Search (Hyperparameter Tuning)

Tested variations of:  
`dropout_rate, batch_norm, skip_connections, bottleneck, hidden_layers, learning_rate, batch_size, epochs`

| Best Model | Dropout | BN | Bottleneck | Skip | LR | F1 | AUC |
|------------|---------|----|------------|------|----|----|-----|
| random_3 | 0.3 | ✅ | ❌ | ✅ | 0.0005 | 0.619 | 0.674 |
| random_0 | 0.4 | ❌ | ❌ | ✅ | 0.001 | 0.624 | 0.672 |
| random_4 | 0.1 | ✅ | ✅ | ❌ | 0.001 | 0.624 | 0.673 |

---

##  8. Best Model Evaluation (on Test Data)

| Metric | Score |
|--------|--------|
| Accuracy | 0.628 |
| Precision | 0.625 |
| Recall | 0.619 |
| F1 Score | 0.632 |
| AUC | 0.668 |

**Confusion Matrix:**

|           | Predicted 0 | Predicted 1 |
|-----------|-------------|-------------|
| True 0    | 11085       | 6323        |
| True 1    | 6473        | 10520       |

🔹 No significant overfitting observed (train and validation losses almost identical).

---

## Week 2 Summary

| Aspect | Result |
|--------|--------|
| Baseline MLP | Accuracy ≈ 0.63, AUC ≈ 0.67 |
| Best Setup | Dropout + Batch Normalization |
| Dataset | Balanced, clean, numerical, normalized |
| Overfitting | Not detected |
| Outcome | Stable baseline ready for Week 3 optimizations |

---
