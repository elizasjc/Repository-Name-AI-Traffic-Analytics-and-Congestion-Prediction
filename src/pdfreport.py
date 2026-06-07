from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet

import pandas as pd
from datetime import datetime
import os

# -----------------------------
# LOAD DATA
# -----------------------------

df = pd.read_csv("data/traffic_counts.csv")

doc = SimpleDocTemplate(
    "outputs/Traffic_Report.pdf"
)

styles = getSampleStyleSheet()

elements = []

# -----------------------------
# TITLE
# -----------------------------

elements.append(
    Paragraph(
        "AI-Powered Smart Traffic Management Report",
        styles["Title"]
    )
)

elements.append(Spacer(1, 12))

elements.append(
    Paragraph(
        f"Generated on: {datetime.now()}",
        styles["BodyText"]
    )
)

elements.append(Spacer(1, 20))

# -----------------------------
# EXECUTIVE SUMMARY
# -----------------------------

elements.append(
    Paragraph(
        "Executive Summary",
        styles["Heading1"]
    )
)

summary = """
This report presents the results of an AI-powered traffic
analytics system built using YOLOv8, OpenCV, Machine Learning,
and Streamlit. The system detects vehicles, analyzes traffic
density, predicts congestion levels, and generates insights
for traffic monitoring.
"""

elements.append(
    Paragraph(
        summary,
        styles["BodyText"]
    )
)

elements.append(Spacer(1, 15))

# -----------------------------
# KPI SECTION
# -----------------------------

elements.append(
    Paragraph(
        "Traffic Statistics",
        styles["Heading1"]
    )
)

elements.append(
    Paragraph(
        f"Total Vehicles Detected: {int(df['total_vehicles'].sum())}",
        styles["BodyText"]
    )
)

elements.append(
    Paragraph(
        f"Average Vehicles Per Frame: {round(df['total_vehicles'].mean(),2)}",
        styles["BodyText"]
    )
)

elements.append(
    Paragraph(
        f"Maximum Vehicles Observed: {int(df['total_vehicles'].max())}",
        styles["BodyText"]
    )
)

elements.append(Spacer(1, 15))

# -----------------------------
# VEHICLE ANALYTICS
# -----------------------------

elements.append(
    Paragraph(
        "Vehicle Distribution",
        styles["Heading1"]
    )
)

elements.append(
    Paragraph(
        f"Cars: {int(df['cars'].sum())}",
        styles["BodyText"]
    )
)

elements.append(
    Paragraph(
        f"Motorcycles: {int(df['motorcycles'].sum())}",
        styles["BodyText"]
    )
)

elements.append(
    Paragraph(
        f"Buses: {int(df['buses'].sum())}",
        styles["BodyText"]
    )
)

elements.append(
    Paragraph(
        f"Trucks: {int(df['trucks'].sum())}",
        styles["BodyText"]
    )
)

elements.append(Spacer(1, 15))

# -----------------------------
# DENSITY ANALYSIS
# -----------------------------

elements.append(
    Paragraph(
        "Traffic Density Analysis",
        styles["Heading1"]
    )
)

density_counts = (
    df["traffic_density"]
    .value_counts()
)

for density, count in density_counts.items():

    elements.append(
        Paragraph(
            f"{density}: {count}",
            styles["BodyText"]
        )
    )

elements.append(Spacer(1, 15))

# -----------------------------
# MODEL COMPARISON
# -----------------------------

if os.path.exists(
    "outputs/model_comparison.csv"
):

    elements.append(
        Paragraph(
            "Machine Learning Model Comparison",
            styles["Heading1"]
        )
    )

    comparison = pd.read_csv(
        "outputs/model_comparison.csv"
    )

    for _, row in comparison.iterrows():

        elements.append(
            Paragraph(
                f"{row['Model']} : {round(row['Accuracy']*100,2)}%",
                styles["BodyText"]
            )
        )

    elements.append(Spacer(1,15))

# -----------------------------
# NEW PAGE
# -----------------------------

elements.append(PageBreak())

# -----------------------------
# CHARTS
# -----------------------------

elements.append(
    Paragraph(
        "Analytics Visualizations",
        styles["Heading1"]
    )
)

elements.append(Spacer(1,10))

if os.path.exists(
    "outputs/traffic_trend.png"
):

    elements.append(
        Paragraph(
            "Traffic Trend",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            "outputs/traffic_trend.png",
            width=400,
            height=250
        )
    )

if os.path.exists(
    "outputs/vehicle_distribution.png"
):

    elements.append(
        Paragraph(
            "Vehicle Distribution",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            "outputs/vehicle_distribution.png",
            width=400,
            height=250
        )
    )

if os.path.exists(
    "outputs/confusion_matrix.png"
):

    elements.append(
        Paragraph(
            "Confusion Matrix",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            "outputs/confusion_matrix.png",
            width=400,
            height=250
        )
    )

if os.path.exists(
    "outputs/feature_importance.png"
):

    elements.append(
        Paragraph(
            "Feature Importance Analysis",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            "outputs/feature_importance.png",
            width=400,
            height=250
        )
    )

    elements.append(Spacer(1, 10))

if os.path.exists(
    "outputs/model_comparison.png"
):

    elements.append(
        Paragraph(
            "Model Performance Comparison",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            "outputs/model_comparison.png",
            width=400,
            height=250
        )
    )

    elements.append(Spacer(1, 10))

if os.path.exists(
    "outputs/correlation_heatmap.png"
):

    elements.append(
        Paragraph(
            "Traffic Feature Correlation Analysis",
            styles["Heading2"]
        )
    )

    elements.append(
        Image(
            "outputs/correlation_heatmap.png",
            width=400,
            height=250
        )
    )

    elements.append(Spacer(1, 10))
    
# -----------------------------
# BUILD PDF
# -----------------------------

doc.build(elements)

print(
    "Professional PDF report generated successfully!"
)