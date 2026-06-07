import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart Traffic AI Dashboard",
    page_icon="🚦",
    layout="wide"
)

# -----------------------------
# LOAD DATA
# -----------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Traffic Dataset",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success(
        "Custom dataset loaded successfully!"
    )

else:
    df = pd.read_csv(
        "data/traffic_counts.csv"
    )

# -----------------------------
# TITLE
# -----------------------------
st.title("🚦 AI-Powered Smart Traffic Dashboard")
st.markdown(
    "Computer Vision + Machine Learning based Traffic Analytics Platform"
)
st.sidebar.title("🚦 Dashboard Navigation")

st.sidebar.markdown("""
### 📊 Analytics

- Traffic Overview
- Dataset Preview
- Traffic Trend
- Vehicle Distribution
- Vehicle Share Analysis
- Traffic Density Analysis
- Traffic Scatter Plot

### 🤖 Machine Learning

- Model Comparison
- Confusion Matrix
- Feature Importance
- Congestion Prediction

### 📄 Reporting

- PDF Report Download

### 📂 Data

- CSV Upload
""")
# -----------------------------
# KPI SECTION
# -----------------------------
st.subheader("Traffic Overview")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Total Vehicles",
    int(df["total_vehicles"].sum())
)

col2.metric(
    "Average Vehicles",
    round(df["total_vehicles"].mean(), 2)
)

col3.metric(
    "Maximum Vehicles",
    int(df["total_vehicles"].max())
)

# -----------------------------
# DATASET PREVIEW
# -----------------------------
st.subheader("Dataset Preview")

st.dataframe(df.head())

st.markdown("---")
st.header("🚗 Vehicle Analytics")

# -----------------------------
# TRAFFIC TREND
# -----------------------------
st.subheader("Traffic Trend Over Time")

st.line_chart(
    df.set_index("frame")["total_vehicles"]
)

# -----------------------------
# VEHICLE DISTRIBUTION BAR CHART
# -----------------------------
vehicle_totals = {
    "Cars": df["cars"].sum(),
    "Motorcycles": df["motorcycles"].sum(),
    "Buses": df["buses"].sum(),
    "Trucks": df["trucks"].sum()
}

vehicle_df = pd.DataFrame(
    vehicle_totals.items(),
    columns=["Vehicle", "Count"]
)

st.subheader("Vehicle Distribution")

st.bar_chart(
    vehicle_df.set_index("Vehicle")
)

# -----------------------------
# PIE CHART 1
# VEHICLE TYPE SHARE
# -----------------------------
st.subheader("Vehicle Type Share")

fig1, ax1 = plt.subplots()

ax1.pie(
    vehicle_df["Count"],
    labels=vehicle_df["Vehicle"],
    autopct="%1.1f%%"
)

ax1.set_title(
    "Vehicle Share"
)

st.pyplot(fig1)

# -----------------------------

import matplotlib.pyplot as plt

numeric_df = df[
    [
        "cars",
        "motorcycles",
        "buses",
        "trucks",
        "total_vehicles"
    ]
]

corr = numeric_df.corr()

fig, ax = plt.subplots(
    figsize=(6,5)
)

im = ax.imshow(corr)

ax.set_xticks(
    range(len(corr.columns))
)

ax.set_yticks(
    range(len(corr.columns))
)

ax.set_xticklabels(
    corr.columns,
    rotation=45
)

ax.set_yticklabels(
    corr.columns
)

plt.colorbar(im)

st.subheader(
    "Feature Correlation Heatmap"
)

plt.savefig(
    "outputs/correlation_heatmap.png",
    bbox_inches="tight"
)

st.pyplot(fig)

st.markdown("---")
st.header("🚦 Traffic Density Analysis")

# -----------------------------
# PIE CHART 2
# TRAFFIC DENSITY DISTRIBUTION
# -----------------------------
st.subheader("Traffic Density Distribution")

density_counts = (
    df["traffic_density"]
    .value_counts()
)

fig2, ax2 = plt.subplots()

ax2.pie(
    density_counts.values,
    labels=density_counts.index,
    autopct="%1.1f%%"
)

ax2.set_title(
    "Traffic Density Composition"
)

st.pyplot(fig2)

# -----------------------------
# SCATTER PLOT
# -----------------------------
st.subheader("Traffic Scatter Plot")

fig3, ax3 = plt.subplots(
    figsize=(8,5)
)

ax3.scatter(
    df["frame"],
    df["total_vehicles"]
)

ax3.set_xlabel("Frame")

ax3.set_ylabel(
    "Total Vehicles"
)

ax3.set_title(
    "Vehicle Count Variability"
)

st.pyplot(fig3)

# -----------------------------
# TRAFFIC DENSITY COUNTS
# -----------------------------
st.subheader(
    "Traffic Density Frequency"
)

st.bar_chart(
    density_counts
)

st.markdown("---")
st.header("🤖 Machine Learning Analysis")

# -----------------------------
# MODEL COMPARISON
# -----------------------------
if os.path.exists(
    "outputs/model_comparison.csv"
):

    st.subheader(
        "Model Comparison"
    )

    comparison = pd.read_csv(
        "outputs/model_comparison.csv"
    )

    st.dataframe(
        comparison
    )

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
if os.path.exists(
    "outputs/confusion_matrix.png"
):

    st.subheader(
        "Confusion Matrix"
    )

    st.image(
        "outputs/confusion_matrix.png"
    )

if os.path.exists(
    "outputs/feature_importance.png"
):

    st.subheader(
        "Feature Importance Analysis"
    )

    st.image(
        "outputs/feature_importance.png"
    )

st.markdown("---")
st.header("🧠 Congestion Prediction")

# -----------------------------
# CONGESTION PREDICTION
# -----------------------------
if os.path.exists(
    "models/congestion_model.pkl"
):

    st.subheader(
        "Congestion Prediction"
    )

    model = joblib.load(
        "models/congestion_model.pkl"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        hour = st.slider(
            "Hour",
            0,
            23,
            8
        )

    with col2:

        day = st.slider(
            "Day of Week",
            1,
            7,
            1
        )

    with col3:

        vehicles = st.slider(
            "Vehicle Count",
            0,
            100,
            30
        )

    prediction = model.predict(
        [[hour, day, vehicles]]
    )

    st.success(
        f"Predicted Traffic Density: {prediction[0]}"
    )

st.markdown("---")
st.header("📄 Reporting")

# -----------------------------
# PDF DOWNLOAD
# -----------------------------


# PDF REPORT DOWNLOAD

if os.path.exists(
    "outputs/Traffic_Report.pdf"
):

    st.subheader(
        "Download Analytics Report"
    )

    with open(
        "outputs/Traffic_Report.pdf",
        "rb"
    ) as pdf_file:

        st.download_button(
            label="📄 Download Traffic Report",
            data=pdf_file,
            file_name="Traffic_Report.pdf",
            mime="application/pdf"
        )
        
st.markdown(
    "Built using YOLOv8, OpenCV, Scikit-learn, Streamlit, and Python"
)