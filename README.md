# 🚦 AI-Powered Traffic Analytics and Congestion Prediction Platform

## 📌 Overview

The AI-Powered Traffic Analytics and Congestion Prediction Platform is an end-to-end Data Science and Machine Learning solution that analyzes traffic videos, detects vehicles using Computer Vision, generates traffic datasets, predicts congestion levels, and presents insights through an interactive Streamlit dashboard.

The project combines Computer Vision, Data Analytics, Machine Learning, Explainable AI, and Automated Reporting to support data-driven traffic management and smart transportation initiatives.

---

## 🎯 Problem Statement

Urban traffic congestion leads to increased travel time, fuel consumption, accidents, and environmental pollution.

Traditional traffic monitoring systems often require manual observation and lack predictive capabilities.

This project leverages Artificial Intelligence to:

* Detect and count vehicles automatically
* Analyze traffic density patterns
* Predict congestion levels
* Visualize insights through interactive dashboards
* Generate automated traffic reports

---

## 🚀 Key Features

### Computer Vision

* Vehicle detection using YOLOv8
* Automatic vehicle counting from traffic videos
* Traffic dataset generation

### Data Analytics

* Traffic trend analysis
* Vehicle distribution analysis
* Traffic density classification
* Correlation analysis between traffic features

### Machine Learning

* Congestion prediction using Random Forest Classifier
* Model training and evaluation
* Model comparison across multiple algorithms
* Saved trained model for deployment

### Explainable AI

* Feature Importance Analysis
* Confusion Matrix Visualization
* Model Performance Comparison

### Dashboard & Reporting

* Interactive Streamlit Dashboard
* Real-time congestion prediction interface
* Automated PDF report generation
* PDF report download functionality

---

## 🏗️ System Architecture

Traffic Video

↓

YOLOv8 Vehicle Detection

↓

Vehicle Count Extraction

↓

Traffic Dataset Creation

↓

Exploratory Data Analysis

↓

Machine Learning Models

↓

Congestion Prediction

↓

Streamlit Dashboard

↓

Automated PDF Reporting

---

## 🛠️ Technology Stack

### Programming Language

* Python

### Computer Vision

* OpenCV
* YOLOv8 (Ultralytics)

### Machine Learning

* Scikit-Learn
* Random Forest Classifier

### Data Processing

* Pandas
* NumPy

### Data Visualization

* Matplotlib

### Dashboard Development

* Streamlit

### Reporting

* ReportLab

### Development Environment

* Visual Studio Code
* Git & GitHub

---

## 📊 Dashboard Components

### Traffic Analytics

* Traffic Overview Metrics
* Dataset Preview
* Traffic Trend Visualization
* Vehicle Distribution Analysis
* Vehicle Share Pie Chart
* Traffic Density Analysis

### Machine Learning Analysis

* Model Comparison
* Confusion Matrix
* Feature Importance Analysis
* Correlation Heatmap

### Prediction Module

* Interactive Congestion Prediction
* User Input Controls

### Reporting Module

* Automated PDF Generation
* Downloadable Traffic Reports

---

## 📈 Results

The system successfully:

* Detects vehicles from traffic videos
* Generates traffic datasets automatically
* Predicts congestion levels using Machine Learning
* Provides explainable AI insights
* Visualizes traffic patterns interactively
* Generates professional PDF reports

---

## 📂 Project Structure

smarttrafficai/

├── app.py

├── README.md

├── data/

│ └── traffic_counts.csv

├── models/

│ └── congestion_model.pkl

├── outputs/

│ ├── confusion_matrix.png

│ ├── feature_importance.png

│ ├── correlation_heatmap.png

│ ├── model_comparison.png

│ ├── Traffic_Report.pdf

│ └── traffic_trend.png

├── src/

│ ├── vehicle_detection.py

│ ├── congestion_prediction.py

│ ├── model_comparison.py

│ └── pdf_report.py

└── videos/

---

## ▶️ Installation

Clone the repository:

```bash
git clone <repository-url>
cd smarttrafficai
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 🔮 Future Enhancements

* Real-time CCTV integration
* Cloud deployment
* Traffic signal optimization
* Weather-aware congestion prediction
* Smart city integration
* Live traffic monitoring dashboards

---

## 👩‍💻 Author

Shilpa J Chethalen

Data Science | Machine Learning | AI Analytics
