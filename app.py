import streamlit as st
import pandas as pd

# Set page layout and configuration
st.set_page_config(page_title="Parkinson's Preprocessing Benchmark", layout="wide", page_icon="🧠")

st.title("🧠 Parkinson's Disease Diagnostic Analysis: A Preprocessing Study")
st.markdown("""
This interactive portfolio showcases a benchmark comparison of **7 Machine Learning Algorithms** across three distinct data processing scenarios on voice diagnostic metrics. 

Rather than just training models, this case study explicitly highlights the significant real-world performance discrepancies caused by **Data Leakage** (oversampling before validation splitting) versus **Dimensionality Loss** (PCA).
""")

st.header("🚀 Preprocessing Pipelines & Methodological Insights")
col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("❌ 1. Naive Workflow (Data Leakage)")
    st.markdown("""
    * **Method:** SMOTE was applied globally prior to splitting dataset.
    * **Result:** Deceptively high model scores (96%+). Synthetic points derived from the test observations leaked back into training, over-optimistically boosting performance.
    """)
with col2:
    st.subheader("🛡️ 2. Corrected Workflow (Real-world)")
    st.markdown("""
    * **Method:** Train/Test split executed first; SMOTE oversampling restricted *only* to the training subset.
    * **Result:** Realistic, generalizable results (XGBoost leading with 92.3% Accuracy and 96.4% Precision).
    """)
with col3:
    st.subheader("📉 3. Principal Component Analysis (PCA)")
    st.markdown("""
    * **Method:** Applied PCA retaining 95% variance on the global dataset, resulting in 3 principal components.
    * **Result:** Consistent reduction in accuracy across classifiers. Reducing high-dimensional clinical voice signatures down to 3 principal components causes information loss that degrades diagnostic performance.
    """)

st.markdown("---")
st.header("📊 Comprehensive Algorithm Benchmarks")

# Scenario 1 Table
st.subheader("Scenario A: Global SMOTE (With Active Data Leakage)")
leakage_df = pd.DataFrame({
    'Algorithm': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'KNN', 'XGBoost'],
    'Accuracy': [0.9322, 0.9661, 0.8305, 0.9661, 0.7627, 0.9661, 0.9322],
    'Precision': [0.9583, 0.9615, 0.9000, 1.0000, 0.9286, 1.0000, 0.8929],
    'Recall': [0.8846, 0.9615, 0.6923, 0.9231, 0.5000, 0.9231, 0.9615],
    'F1-Score': [0.9200, 0.9615, 0.7826, 0.9600, 0.6500, 0.9600, 0.9259]
})
st.table(leakage_df)

# Scenario 2 Table
st.subheader("Scenario B: Correct SMOTE (No Leakage - Model Generalization)")
correct_df = pd.DataFrame({
    'Algorithm': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'KNN', 'XGBoost'],
    'Accuracy': [0.8718, 0.8974, 0.7949, 0.8205, 0.6667, 0.8718, 0.9231],
    'Precision': [0.9615, 0.9310, 1.0000, 1.0000, 1.0000, 1.0000, 0.9643],
    'Recall': [0.8621, 0.9310, 0.7241, 0.7586, 0.5517, 0.8276, 0.9310],
    'F1-Score': [0.9091, 0.9310, 0.8400, 0.8627, 0.7111, 0.9057, 0.9474]
})
st.table(correct_df)

# Scenario 3 Table
st.subheader("Scenario C: Global SMOTE + PCA (Dimensionality Reduced - 3 Components)")
pca_df = pd.DataFrame({
    'Algorithm': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'KNN', 'XGBoost'],
    'Accuracy': [0.7797, 0.8814, 0.7627, 0.9153, 0.7458, 0.8644, 0.8305],
    'Precision': [0.7241, 0.8519, 0.6765, 0.8387, 0.6667, 0.8462, 0.7857],
    'Recall': [0.8077, 0.8846, 0.8846, 1.0000, 0.8462, 0.8462, 0.8462],
    'F1-Score': [0.7636, 0.8679, 0.7667, 0.9123, 0.7458, 0.8462, 0.8148]
})
st.table(pca_df)

st.markdown("""
---
### 🔬 Key Takeaway Conclusions:
1. **Data Leakage is highly deceptive**: The 96.6% accuracy from naive oversampling is falsified. Our true, honest generalization benchmark shows **XGBoost** (Scenario B) as the most robust predictor with **92.31% Accuracy**.
2. **PCA Information Loss**: When applying PCA to highly descriptive voice diagnostic variables on small datasets, performance degrades. In this case, keeping the physical 22 audio features yields superior class separability than reducing down to 3 latent principal components.
""")
