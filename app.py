import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set page layout and configuration
st.set_page_config(page_title="Parkinson's Detection App", layout="wide", page_icon="🧠")

# Load Saved Scaler and All 7 Models
@st.cache_resource
def load_assets():
    scaler = joblib.load('scaler.pkl')
    models = {
        'Decision Tree': joblib.load('dt_model.pkl'),
        'Random Forest': joblib.load('rf_model.pkl'),
        'Logistic Regression': joblib.load('lr_model.pkl'),
        'SVM': joblib.load('svm_model.pkl'),
        'Naive Bayes': joblib.load('nb_model.pkl'),
        'KNN': joblib.load('knn_model.pkl'),
        'XGBoost': joblib.load('xgb_model.pkl')
    }
    return scaler, models

try:
    scaler, models = load_assets()
except Exception as e:
    st.error(f"Error loading assets: {e}. Please make sure scaler.pkl and all model .pkl files are in the directory.")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:", ["Project Overview & Journey", "Interactive Predictor"])

# -----------------------------------------------------
# Tab 1: Project Overview & Journey
# -----------------------------------------------------
if app_mode == "Project Overview & Journey":
    st.title("🧠 Parkinson's Disease Diagnostic Analysis")
    st.markdown("""
    Welcome! This Streamlit app serves as an interactive portfolio showcase for a **Parkinson's Disease Detection** system using clinical voice recordings.
    Instead of just training a model, this project explores a deep journey from **identifying crucial methodology errors (Data Leakage)** to comparing dimensionality reduction techniques.
    """)

    st.header("🚀 The Engineering Journey & Methodological Corrections")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("❌ 1. The Naive Approach (Data Leakage)")
        st.markdown("""
        * **What happened:** SMOTE was initially applied to the **entire dataset** prior to splitting it into Train/Test subsets.
        * **The Problem:** This caused synthetic records in the test set to be calculated from neighbors in the training set (and vice-versa). This introduces severe **Data Leakage**, resulting in deceptively high accuracy scores that fail in production.
        """)
    with col2:
        st.subheader("✅ 2. The Correct Implementation")
        st.markdown("""
        * **The Fix:** We split the dataset into original Train/Test splits **first**. 
        * **Method:** We applied SMOTE **only to the Training portion** so that the Test set remained untouched, pristine, and structurally representative of real-world clinical data.
        """)

    st.header("📊 Comparative Performance Showcase")
    
    tab1, tab2, tab3 = st.tabs(["🔄 1. With Data Leakage (Original)", "🛡️ 2. Safe & Corrected SMOTE (Live Models)", "📉 3. PCA vs Non-PCA (Dimensionality Impact)"])

    with tab1:
        st.subheader("Results with SMOTE Before Split (Data Leakage)")
        st.markdown("These results looked incredibly strong (e.g., Random Forest at 97% accuracy), but were overly optimistic due to leakage.")
        leakage_data = {
            'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'KNN', 'XGBoost'],
            'Accuracy': [0.932, 0.966, 0.830, 0.966, 0.762, 0.966, 0.932],
            'Precision': [0.958, 0.961, 0.900, 1.000, 0.928, 1.000, 0.892],
            'Recall': [0.884, 0.961, 0.692, 0.923, 0.500, 0.923, 0.961],
            'F1 Score': [0.920, 0.961, 0.782, 0.960, 0.650, 0.960, 0.925]
        }
        st.table(pd.DataFrame(leakage_data))

    with tab2:
        st.subheader("Corrected Workflow: Splitting Before SMOTE")
        st.markdown("Applying SMOTE to training data only ensures complete generalization. Notice that XGBoost emerges as a highly robust predictor.")
        correct_data = {
            'Model': ['Decision Tree', 'Random Forest', 'Logistic Regression', 'SVM', 'Naive Bayes', 'KNN', 'XGBoost'],
            'Accuracy': [0.8718, 0.8974, 0.7949, 0.8205, 0.6667, 0.8718, 0.9231],
            'Precision': [0.9615, 0.9310, 1.0000, 1.0000, 1.0000, 1.0000, 0.9643],
            'Recall': [0.8621, 0.9310, 0.7241, 0.7586, 0.5517, 0.8276, 0.9310],
            'F1-Score': [0.9091, 0.9310, 0.8400, 0.8627, 0.7111, 0.9057, 0.9474]
        }
        st.table(pd.DataFrame(correct_data))

    with tab3:
        st.subheader("Dimensionality Reduction: The Impact of PCA")
        st.markdown("We applied PCA (explaining 95% variance) to 3 components. Because our dataset is relatively small, PCA lost meaningful local variance, reducing accuracy across all classifiers.")
        pca_data = {
            'Model': ['Random Forest (Original)', 'Random Forest (PCA)', 'XGBoost (Original)', 'XGBoost (PCA)'],
            'Accuracy': [0.966, 0.881, 0.932, 0.830],
            'Precision': [0.966, 0.882, 0.934, 0.833],
            'Recall': [0.966, 0.881, 0.932, 0.830],
            'F1 Score': [0.966, 0.881, 0.932, 0.830]
        }
        st.table(pd.DataFrame(pca_data))

# -----------------------------------------------------
# Tab 2: Interactive Predictor Playground
# -----------------------------------------------------
elif app_mode == "Interactive Predictor":
    st.title("🎙️ Parkinson's Interactive Diagnostic Playground")
    st.markdown("""
    Test all 7 trained models live! Adjust the sliders representing clinical voice measurements 
    (vocal fundamental frequency, jitter, shimmer, etc.) to see how different model architectures behave.
    """)

    # Default standard values based on UCI dataset distributions
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Vocal Frequencies (Hz)")
        fo = st.slider("MDVP:Fo (Average vocal frequency)", 88.0, 260.0, 154.0)
        fhi = st.slider("MDVP:Fhi (Maximum vocal frequency)", 102.0, 592.0, 197.0)
        flo = st.slider("MDVP:Flo (Minimum vocal frequency)", 65.0, 239.0, 116.0)
        
        st.markdown("### Noise Measures")
        nhr = st.slider("NHR (Noise-to-Harmonics Ratio)", 0.0006, 0.314, 0.024)
        hnr = st.slider("HNR (Harmonics-to-Noise Ratio)", 8.4, 33.0, 21.8)

    with col2:
        st.markdown("### Frequency Jitter (Variations)")
        jitter = st.slider("MDVP:Jitter (%)", 0.0016, 0.033, 0.006)
        jitter_abs = st.slider("MDVP:Jitter(Abs)", 0.000007, 0.00026, 0.00004)
        rap = st.slider("MDVP:RAP", 0.00068, 0.021, 0.003)
        ppq = st.slider("MDVP:PPQ", 0.0009, 0.019, 0.003)
        ddp = st.slider("Jitter:DDP", 0.002, 0.064, 0.01)

    with col3:
        st.markdown("### Amplitude Shimmer (Variations)")
        shimmer = st.slider("MDVP:Shimmer", 0.009, 0.119, 0.029)
        shimmer_db = st.slider("MDVP:Shimmer(dB)", 0.085, 1.30, 0.28)
        apq3 = st.slider("Shimmer:APQ3", 0.003, 0.056, 0.014)
        apq5 = st.slider("Shimmer:APQ5", 0.005, 0.079, 0.017)
        apq = st.slider("MDVP:APQ", 0.007, 0.137, 0.02)
        dda = st.slider("Shimmer:DDA", 0.013, 0.169, 0.046)

    # More complex features
    st.markdown("### Nonlinear Measures & Voice Variations")
    col_extra1, col_extra2, col_extra3, col_extra4, col_extra5 = st.columns(5)
    with col_extra1:
        rpde = st.slider("RPDE", 0.25, 0.68, 0.49)
    with col_extra2:
        dfa = st.slider("DFA", 0.57, 0.82, 0.71)
    with col_extra3:
        spread1 = st.slider("spread1", -7.9, -2.4, -5.6)
    with col_extra4:
        spread2 = st.slider("spread2", 0.0, 0.45, 0.22)
    with col_extra5:
        d2 = st.slider("D2", 1.4, 3.6, 2.3)
    
    ppe = st.slider("PPE", 0.04, 0.52, 0.20)

    # Model Selection for Playground - All 7 models listed!
    st.markdown("---")
    selected_model_name = st.selectbox(
        "Choose Prediction Model:", 
        ["XGBoost", "Random Forest", "Decision Tree", "Logistic Regression", "SVM", "Naive Bayes", "KNN"]
    )
    
    # Construct input vector
    input_features = np.array([[fo, fhi, flo, jitter, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db,
                                apq3, apq5, apq, dda, nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])

    # Perform Scaling using loaded MinMaxScaler
    scaled_input = scaler.transform(input_features)

    # Run Prediction
    if st.button("🔍 Run Voice Diagnostic Prediction"):
        model_to_use = models[selected_model_name]
        prediction = model_to_use.predict(scaled_input)[0]
        
        # Some classifiers (like custom SVM instances without probability=True) might fail on predict_proba,
        # but we set probability=True on our models.
        try:
            prediction_prob = model_to_use.predict_proba(scaled_input)[0]
            conf = f" (Confidence: {prediction_prob[int(prediction)]*100:.2f}%)"
        except:
            conf = ""

        if prediction == 1:
            st.error(f"🚨 **Result: Positive for Parkinson's Disease**{conf}")
        else:
            st.success(f"💚 **Result: Healthy / Non-Parkinson's**{conf}")
