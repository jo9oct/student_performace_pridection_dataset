# app/streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# --------------------------------------------------
# App Configuration
# --------------------------------------------------
st.set_page_config(
    page_title="🎓 Student Performance Analytics & Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎓"
)

st.markdown("<h1 style='text-align:center;'>🎓 Student Performance Analytics & Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Explore & predict student performance with interactive analytics</p>", unsafe_allow_html=True)
st.divider()

# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------
@st.cache_resource
def load_model():
    with open("../models/model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_preprocessing():
    with open("../models/preprocessing.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("../data/raw/student_performance_dataset.csv")

model = load_model()
preprocessing_pipeline = load_preprocessing()
df = load_data()

# --------------------------------------------------
# Sidebar Navigation
# --------------------------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Interactive EDA", "Make Prediction"]
)

# ==================================================
# PAGE 1: INTERACTIVE EDA
# ==================================================
if page == "Interactive EDA":

    st.header("📊 Interactive Exploratory Data Analysis")

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    st.subheader("Univariate Analysis")
    selected_num = st.selectbox("Select a numerical feature", numeric_cols)

    fig, ax = plt.subplots()
    sns.histplot(df[selected_num], kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_num}")
    st.pyplot(fig)

    st.subheader("Categorical Feature Distribution")
    selected_cat = st.selectbox("Select a categorical feature", categorical_cols)

    fig, ax = plt.subplots()
    df[selected_cat].value_counts().plot(kind="bar", ax=ax)
    ax.set_title(f"Distribution of {selected_cat}")
    st.pyplot(fig)

    st.subheader("Correlation Analysis")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ==================================================
# PAGE 2: PREDICTION (SLIDER-BASED UI)
# ==================================================
elif page == "Make Prediction":

    st.header("🔮 Student Pass / Fail Prediction")
    st.write("Adjust the sliders and inputs in the sidebar to predict student performance.")

    # Drop unused columns
    input_df = df.drop(columns=["Student_ID", "Final_Exam_Score", "Pass_Fail"], errors="ignore")

    # --------------------------------
    # SIDEBAR INPUTS
    # --------------------------------
    st.sidebar.subheader("📋 Student Inputs")

    user_input = {}

    for col in input_df.columns:

        # Numeric sliders
        if input_df[col].dtype in ["int64", "float64"]:

            min_val = int(input_df[col].min())
            max_val = int(input_df[col].max())
            default_val = int(input_df[col].median())

            # Custom emoji labels
            label = col.replace("_", " ").title()

            user_input[col] = st.sidebar.slider(
                f"📊 {label}",
                min_value=min_val,
                max_value=max_val,
                value=default_val
            )

        # Categorical dropdowns
        else:
            user_input[col] = st.sidebar.selectbox(
                f"📂 {col.replace('_', ' ').title()}",
                options=input_df[col].unique().tolist()
            )

    input_data = pd.DataFrame([user_input])

    # --------------------------------
    # MAIN PREDICTION PANEL
    # --------------------------------
    if st.button("🎯 Predict", use_container_width=True):

        input_processed = preprocessing_pipeline.transform(input_data)
        probability = model.predict_proba(input_processed)[0][1]
        prediction = 1 if probability >= 0.45 else 0

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Prediction",
            "PASS ✅" if prediction == 1 else "FAIL ❌",
            delta=f"Confidence {probability:.2f}"
        )

        col2.metric("Pass Probability", f"{probability * 100:.2f}%")

        risk = (
            "Low Risk" if probability >= 0.7
            else "Medium Risk" if probability >= 0.45
            else "High Risk"
        )

        col3.metric("Risk Level", risk)

        st.progress(float(probability))

        if prediction == 1:
            st.success(f"✅ The student is likely to PASS")
        else:
            st.error(f"⚠️ The student is at risk of FAILING")

        # --------------------------------
        # Risk Gauge
        # --------------------------------
        st.subheader("📊 Pass Probability Gauge")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={'text': "Pass Probability (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'steps': [
                    {'range': [0, 45], 'color': "#FF6F6F"},
                    {'range': [45, 70], 'color': "#FFDD57"},
                    {'range': [70, 100], 'color': "#4ADE80"}
                ]
            }
        ))

        st.plotly_chart(fig, use_container_width=True)

        # --------------------------------
        # Feature Overview
        # --------------------------------
        st.subheader("📊 Input Feature Overview")

        features = input_data.columns.tolist()
        values = [
            v if isinstance(v, (int, float)) else 1
            for v in input_data.iloc[0].values
        ]

        fig2 = go.Figure(go.Bar(
            x=features,
            y=values
        ))

        fig2.update_layout(
            title="Student Input Features",
            yaxis_title="Value"
        )

        st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    "<hr><p style='text-align:center;color:gray;'>📘 Machine Learning Project | Streamlit Dashboard 🚀</p>",
    unsafe_allow_html=True
)
