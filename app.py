
import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Heart AI | Heart Disease Classification",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================================================
# 2. LOAD MACHINE LEARNING MODEL
# ==================================================

@st.cache_resource
def load_model():
    return joblib.load("heart_disease_logistic_model.joblib")


model = load_model()

# ==================================================
# 3. PROFESSIONAL DESIGN & CUSTOM CSS
# ==================================================

st.markdown("""
<style>
.stApp {
    font-family: 'Segoe UI', sans-serif;
}

footer {
    visibility: hidden;
}

.block-container {
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 38px;
    font-weight: 800;
    color: #2563EB;
    margin-bottom: 5px;
}

.subtitle {
    font-size: 16px;
    color: #64748b;
    margin-bottom: 25px;
}

.section-title {
    font-size: 25px;
    font-weight: 750;
    margin-top: 25px;
    margin-bottom: 15px;
}

.info-card {
    padding: 22px;
    border-radius: 14px;
    background-color: rgba(100, 116, 139, 0.08);
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin-bottom: 15px;
}

.disclaimer {
    padding: 18px;
    border-radius: 12px;
    background-color: rgba(245, 158, 11, 0.10);
    border-left: 5px solid #D97706;
    margin-top: 20px;
}

.stButton > button,
.stFormSubmitButton > button {
    border-radius: 9px;
    font-weight: 650;
    min-height: 43px;
}

[data-testid="stMetric"] {
    background-color: rgba(100, 116, 139, 0.07);
    border: 1px solid rgba(148, 163, 184, 0.20);
    padding: 18px;
    border-radius: 12px;
}

section[data-testid="stSidebar"] {
    border-right: 1px solid rgba(148, 163, 184, 0.20);
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# 4. WEBSITE HEADER
# ==================================================

st.html("""
<div style="
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 20px;
    padding: 24px 30px;
    margin-bottom: 30px;
    border-radius: 16px;
    background: linear-gradient(110deg, #102A43, #1D4ED8);
    color: white;
    box-shadow: 0 5px 18px rgba(15, 23, 42, 0.15);
">

    <div>
        <div style="
            font-size: 28px;
            font-weight: 850;
            letter-spacing: 1px;
        ">
            HEART <span style="color:#93C5FD;">AI</span>
        </div>

        <div style="
            font-size: 11px;
            letter-spacing: 1.5px;
            opacity: 0.85;
            margin-top: 4px;
        ">
            HEART DISEASE CLASSIFICATION
        </div>
    </div>

    <div style="
        text-align: right;
        font-family: 'Segoe UI', sans-serif;
    ">
        <div style="
            font-size: 10px;
            letter-spacing: 1.5px;
            opacity: 0.8;
        ">
            DEVELOPED BY
        </div>

        <div style="
            font-size: 20px;
            font-weight: 750;
            margin-top: 4px;
        ">
            Hareth Soboh
        </div>

        <div style="
            font-size: 11px;
            opacity: 0.8;
            margin-top: 3px;
        ">
            Machine Learning Project
        </div>
    </div>

</div>

<style>
@media (max-width: 600px) {
    div[style*="linear-gradient(110deg"] {
        flex-direction: column;
        align-items: flex-start !important;
    }

    div[style*="text-align: right"] {
        text-align: left !important;
    }
}
</style>
""")

# ==================================================
# 5. SIDEBAR NAVIGATION
# ==================================================

st.sidebar.markdown("""
<div style="padding: 10px 0 15px 0;">
    <div style="font-size:25px;font-weight:850;color:#2563EB;">
        ❤️ HEART AI
    </div>
    <div style="font-size:11px;color:gray;">
        MACHINE LEARNING PROJECT
    </div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "MAIN MENU",
    [
        "Home",
        "Interactive Prediction",
        "Model Performance",
        "Methodology",
        "About & Limitations"
    ]
)

st.sidebar.divider()

st.sidebar.markdown("### Project Information")

st.sidebar.caption("""
An academic machine learning project exploring
binary classification using the Cleveland
Heart Disease dataset.
""")

st.sidebar.divider()

st.sidebar.markdown("**Developed by**")
st.sidebar.write("Hareth Soboh")
st.sidebar.caption("Academic Project | 2026")

# ==================================================
# 6. HOME PAGE
# ==================================================

if page == "Home":

    st.markdown(
        '<div class="main-title">Heart Disease Classification</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">'
        'An Interactive Machine Learning Classification Platform'
        '</div>',
        unsafe_allow_html=True
    )

    st.write("""
    This project explores how machine learning algorithms can
    classify records in the Cleveland Heart Disease dataset.

    Five machine learning algorithms were trained, evaluated,
    compared, and optimized using cross-validation and
    hyperparameter tuning.

    The application provides an interactive demonstration
    of the selected Logistic Regression model.
    """)

    st.divider()

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "297")
    col2.metric("Input Features", "13")
    col3.metric("Algorithms Compared", "5")

    st.markdown(
        '<div class="section-title">Project Objectives</div>',
        unsafe_allow_html=True
    )

    objective1, objective2 = st.columns(2)

    with objective1:

        st.markdown("""
        <div class="info-card">
        <b>Data Analysis</b><br><br>
        Explore the dataset, understand its features,
        and investigate patterns in the data.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
        <b>Model Development</b><br><br>
        Train and compare five machine learning
        classification algorithms.
        </div>
        """, unsafe_allow_html=True)

    with objective2:

        st.markdown("""
        <div class="info-card">
        <b>Model Evaluation</b><br><br>
        Apply classification metrics, cross-validation,
        and hyperparameter tuning.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="info-card">
        <b>Interactive Application</b><br><br>
        Build a user-friendly web application
        to demonstrate the selected model.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer">
    <b>Important Notice</b><br><br>
    This application is an educational machine learning
    demonstration. It is not a medical device and must
    not be used for diagnosis or healthcare decisions.
    </div>
    """, unsafe_allow_html=True)

# ==================================================
# 7. INTERACTIVE PREDICTION PAGE
# ==================================================

elif page == "Interactive Prediction":

    st.title("Interactive Classification")

    st.write("""
    Enter the 13 input features to see how the trained
    machine learning model classifies the record.
    """)

    st.info("""
    The input ranges reflect the observed values in the dataset.
    They are not clinical reference ranges.
    """)

    with st.form("prediction_form"):

        st.subheader("01 | Basic Information")

        col1, col2 = st.columns(2)

        with col1:

            age = st.slider("Age", 29, 77, 54)

            sex = st.selectbox(
                "Sex",
                [0, 1],
                format_func=lambda x:
                "Female (0)" if x == 0 else "Male (1)"
            )

            cp = st.selectbox(
                "Chest Pain Type (cp)",
                [0, 1, 2, 3]
            )

            trestbps = st.slider(
                "Resting Blood Pressure (trestbps)",
                94, 200, 130
            )

            chol = st.slider(
                "Cholesterol (chol)",
                126, 564, 240
            )

            fbs = st.selectbox(
                "Fasting Blood Sugar (fbs)",
                [0, 1],
                format_func=lambda x:
                "False (0)" if x == 0 else "True (1)"
            )

            restecg = st.selectbox(
                "Resting ECG (restecg)",
                [0, 1, 2]
            )

        with col2:

            thalach = st.slider(
                "Maximum Heart Rate (thalach)",
                71, 202, 150
            )

            exang = st.selectbox(
                "Exercise Induced Angina (exang)",
                [0, 1],
                format_func=lambda x:
                "No (0)" if x == 0 else "Yes (1)"
            )

            oldpeak = st.slider(
                "Oldpeak",
                min_value=0.0,
                max_value=6.2,
                value=1.0,
                step=0.1
            )

            slope = st.selectbox("Slope", [0, 1, 2])

            ca = st.selectbox(
                "Number of Major Vessels (ca)",
                [0, 1, 2, 3]
            )

            thal = st.selectbox("Thal", [0, 1, 2])

        submitted = st.form_submit_button(
            "Run Classification",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal
        }])

        prediction = model.predict(input_data)[0]
        probabilities = model.predict_proba(input_data)[0]

        st.divider()
        st.subheader("Classification Results")

        if prediction == 0:

            st.success("Predicted Dataset Class: 0")
            st.write("The model assigned this record to class 0.")

        else:

            st.warning("Predicted Dataset Class: 1")
            st.write("The model assigned this record to class 1.")

        col1, col2 = st.columns(2)

        col1.metric(
            "Class 0 Probability",
            f"{probabilities[0] * 100:.2f}%"
        )

        col2.metric(
            "Class 1 Probability",
            f"{probabilities[1] * 100:.2f}%"
        )

        st.caption("""
        These values represent model-estimated probabilities
        for dataset classes, not a person's clinical probability
        of disease.
        """)

        st.markdown("""
        <div class="disclaimer">
        <b>Educational Use Only</b><br><br>
        This classification result is not a medical diagnosis.
        </div>
        """, unsafe_allow_html=True)

# ==================================================
# 8. MODEL PERFORMANCE PAGE
# ==================================================

elif page == "Model Performance":

    st.title("Model Performance")

    st.write("""
    Comparison of the five tuned machine learning algorithms
    using the held-out test dataset.
    """)

    results = pd.DataFrame({
        "Algorithm": [
            "MLP",
            "Logistic Regression",
            "Random Forest",
            "SVM",
            "Decision Tree"
        ],
        "Accuracy": [
            0.9333, 0.9167, 0.8833, 0.8833, 0.7500
        ],
        "Precision": [
            1.0000, 1.0000, 1.0000, 1.0000, 0.7600
        ],
        "Recall": [
            0.8571, 0.8214, 0.7500, 0.7500, 0.6786
        ],
        "F1-Score": [
            0.9231, 0.9020, 0.8571, 0.8571, 0.7170
        ]
    })

    st.subheader("Test Set Evaluation")

    st.dataframe(
        results.style.format({
            "Accuracy": "{:.2%}",
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "F1-Score": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    # Horizontal Test Accuracy Comparison Chart

    st.subheader("Test Accuracy Comparison")

    chart_data = results.sort_values(
        "Accuracy",
        ascending=True
    )

    fig = px.bar(
        chart_data,
        x="Accuracy",
        y="Algorithm",
        orientation="h",
        text="Accuracy",
        color="Accuracy",
        color_continuous_scale="Blues",
        title="Test Accuracy Comparison"
    )

    fig.update_traces(
        texttemplate="%{text:.2%}",
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="Accuracy",
        yaxis_title="Algorithm",
        xaxis_tickformat=".0%",
        xaxis_range=[0, 1.08],
        yaxis=dict(categoryorder="total ascending"),
        showlegend=False,
        height=400,
        margin=dict(l=20, r=50, t=60, b=40)
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.subheader("Cross-Validation Accuracy")

    cv_results = pd.DataFrame({
        "Algorithm": [
            "Logistic Regression",
            "SVM",
            "Random Forest",
            "Decision Tree",
            "MLP"
        ],
        "CV Accuracy": [
            0.8231, 0.8145, 0.8105, 0.7598, 0.7471
        ]
    })

    st.dataframe(
        cv_results.style.format({
            "CV Accuracy": "{:.2%}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("ROC-AUC Comparison")

    auc_results = pd.DataFrame({
        "Algorithm": [
            "Logistic Regression",
            "Random Forest",
            "MLP",
            "SVM",
            "Decision Tree"
        ],
        "ROC-AUC": [
            0.9754, 0.9699, 0.9531, 0.9487, 0.8315
        ]
    })

    st.dataframe(
        auc_results.style.format({
            "ROC-AUC": "{:.4f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.info("""
    MLP achieved the highest test accuracy (93.33%).
    Logistic Regression achieved the highest cross-validation
    accuracy (82.31%) and ROC-AUC (0.9754).
    """)

# ==================================================
# 9. METHODOLOGY PAGE
# ==================================================

elif page == "Methodology":

    st.title("Project Methodology")

    st.subheader("Dataset Overview")

    st.write("""
    Dataset: Heart Disease Cleveland UCI

    - Total records: 297
    - Input features: 13
    - Target variable: condition
    - Class 0: No disease label in the dataset
    - Class 1: Disease label in the dataset
    """)

    st.divider()

    st.subheader("Data Preprocessing")

    st.markdown("""
    1. Checked missing values and duplicate records.
    2. Separated input features from the target variable.
    3. Split the data into training and testing sets (80/20).
    4. Used stratified sampling to preserve class proportions.
    5. Applied StandardScaler to numerical features.
    6. Applied OneHotEncoder to categorical features.
    7. Used Scikit-learn Pipelines to avoid data leakage.
    """)

    st.divider()

    st.subheader("Machine Learning Algorithms")

    algorithms = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Support Vector Machine (SVM)",
        "Multi-Layer Perceptron (MLP)"
    ]

    for algorithm in algorithms:
        st.markdown(f"- {algorithm}")

    st.divider()

    st.subheader("Evaluation Techniques")

    st.markdown("""
    - Accuracy
    - Precision
    - Recall
    - F1-Score
    - Confusion Matrix
    - Stratified 5-Fold Cross-Validation
    - GridSearchCV
    - ROC-AUC
    """)

# ==================================================
# 10. ABOUT & LIMITATIONS PAGE
# ==================================================

elif page == "About & Limitations":

    st.title("About This Project")

    st.write("""
    This application was developed as an academic machine
    learning project to demonstrate a binary classification
    workflow using the Cleveland Heart Disease dataset.
    """)

    st.divider()

    st.subheader("Project Limitations")

    st.markdown("""
    - The dataset contains only 297 records.
    - The dataset represents a specific historical source.
    - Test results may vary with different datasets.
    - Model probabilities are not clinical risk estimates.
    - Performance does not establish medical validity.
    - The application is intended for education and demonstration only.
    """)

    st.divider()

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas
    - Scikit-learn
    - NumPy
    - Joblib
    - Streamlit
    - Plotly
    """)

    st.divider()

    st.subheader("Developer")

    st.markdown("""
    **Hareth Soboh**

    Academic Machine Learning Project
    """)

    st.warning("""
    This application is not intended for diagnosis, treatment,
    or any healthcare decision. Always consult qualified
    healthcare professionals for medical concerns.
    """)

# ==================================================
# 11. FOOTER
# ==================================================

st.divider()

st.markdown("""
<div style="
    text-align:center;
    padding:15px;
    color:gray;
    font-size:12px;
">
    <b>HEART AI</b><br>
    Heart Disease Classification Using Machine Learning<br>
    Developed by Hareth Soboh | 2026<br><br>
    Educational Project — Not for Medical Use
</div>
""", unsafe_allow_html=True)

st.sidebar.divider()
st.sidebar.caption("HEART AI | 2026")
