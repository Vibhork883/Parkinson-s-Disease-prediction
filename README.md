# 🧠 Parkinson's Disease Prediction using Machine Learning

A machine learning project that predicts whether a person has Parkinson's disease based on biomedical voice measurements. Multiple classification algorithms are trained and compared to find the best-performing model.

> ⚠️ **Disclaimer:** This project is for educational purposes only. It is **not** a medical diagnostic tool and should never be used as a substitute for professional medical advice.

---

## 📌 Project Overview

Parkinson's disease is a progressive neurological disorder that affects movement and speech. Research has shown that voice recordings can capture subtle biomarkers of the disease. This project uses a dataset of voice measurements to build and evaluate several ML models that classify a person as healthy or affected by Parkinson's.

**Workflow followed in this project:**
1. Data collection & exploration
2. Data preprocessing (dropping redundant columns, checking for duplicates/nulls)
3. Exploratory Data Analysis (EDA) — class balance, correlation heatmap, box plots
4. Handling class imbalance using **SMOTE**
5. Feature scaling (MinMaxScaler) and optional dimensionality reduction (**PCA**)
6. Training and tuning 7 classification models
7. Comparing models using Accuracy, Precision, Recall, F1-score, and R² score

---

## 📊 Dataset

- **Source:** [UCI Machine Learning Repository — Parkinson's Disease Dataset](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data)
- **Instances:** 195 voice recordings from 31 people (23 with Parkinson's, 8 healthy)
- **Features:** 22 biomedical voice measurements (e.g., `MDVP:Fo`, `MDVP:Jitter`, `HNR`, `RPDE`, `DFA`, `spread1`, `spread2`, `PPE`, etc.)
- **Target:** `status` → `1` = Parkinson's, `0` = healthy
- File: [`Parkinsons_data.csv`](./Parkinsons_data.csv)

---

## 🤖 Models Trained

| Model | Notes |
|---|---|
| Decision Tree Classifier | Tuned with GridSearchCV |
| Random Forest Classifier | Tuned with GridSearchCV — **best performer** |
| Logistic Regression | Baseline linear model |
| Support Vector Machine (SVM) | Linear & RBF kernels, tuned |
| Naive Bayes (Gaussian) | Baseline probabilistic model |
| K-Nearest Neighbors (KNN) | Tuned over multiple K values |
| XGBoost Classifier | Tuned with GridSearchCV |

Each model was also tested after applying **PCA** for dimensionality reduction, to compare performance with and without it.

---

## 🏆 Results

Best-performing model on the test set:

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| **Random Forest** | **96.6%** | 96.1% | 96.1% | 96.1% |
| XGBoost | 93.2% | 93.5% | 93.2% | 93.2% |
| Decision Tree | 93.2% | 95.8% | 88.5% | 92.0% |
| SVM | 92.0% | — | — | 92.0% |

📉 **Key finding:** PCA actually *hurt* performance for every model in this project. This is expected on small datasets (195 rows) — PCA works best on large, high-dimensional data where dimensionality reduction reduces overfitting; here it just throws away useful information.

---

## 📁 Repository Structure

```
parkinsons-disease-prediction/
├── Parkinsons_Project.ipynb   # Full notebook: EDA, preprocessing, model training & evaluation
├── Parkinsons_data.csv        # Dataset used for training/testing
├── requirements.txt           # Python dependencies
├── .gitignore
└── README.md
```

---

## 🛠️ Tech Stack

- **Language:** Python
- **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn (SMOTE), XGBoost, seaborn, matplotlib
- **Environment:** Jupyter Notebook

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/parkinsons-disease-prediction.git
cd parkinsons-disease-prediction
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the notebook
```bash
jupyter notebook Parkinsons_Project.ipynb
```

---

## 🔭 Future Work

This is the first version of the project. Planned next steps:
- [ ] Export the best model (Random Forest) as a `.pkl` file for inference
- [ ] Build a simple prediction API/backend
- [ ] Deploy a live web app (e.g., on Vercel) so users can input voice-measurement values and get a prediction
- [ ] Add a database (e.g., Supabase) to store user submissions/results
- [ ] Add authentication (e.g., email-based) for saving personal history

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgements

- Dataset: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data)
- Original dataset creators: Max Little of the University of Oxford, in collaboration with the National Centre for Voice and Speech, Denver, Colorado.
