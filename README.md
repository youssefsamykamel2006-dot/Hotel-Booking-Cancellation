# 🏨 Hotel Booking Cancellation Dashboard

## 📌 Project Overview

This project predicts whether a hotel booking is likely to be cancelled using Machine Learning.

The project also includes an interactive Streamlit dashboard that allows users to:

- Explore hotel booking data
- Analyze cancellation trends
- Predict booking cancellations
- View project information

---

## 📊 Dashboard Features

### Dashboard
- Booking KPIs
- Hotel booking distribution
- Cancellation distribution
- Monthly bookings
- Lead time distribution
- Business insights

### Analysis
- Interactive filters
- Cancellation analysis
- ADR analysis
- Lead Time analysis
- Market Segment analysis
- Deposit Type analysis
- Correlation Heatmap

### Prediction
Predict whether a booking will be cancelled using the trained XGBoost model.

### About
Project information and model performance.

---

## 🤖 Machine Learning

Best Model:

**XGBoost Classifier**

Performance:

| Metric | Score |
|---------|-------|
| Accuracy | **85.6%** |
| Precision | **74.8%** |
| Recall | **71.9%** |
| F1 Score | **73.3%** |

---

## 📁 Project Structure

```
Hotel-Booking-Cancellation/
│
├── app/
│   └── app.py
│
├── data/
│   └── hotel_bookings_final.csv
│
├── models/
│   └── best_xgboost_model.pkl
│
├── notebooks/
│
├── requirements.txt
│
└── README.md
```

---

## ⚙️ Technologies

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-Learn
- XGBoost
- Imbalanced-Learn

---

## 🚀 Run Locally

```bash
pip install -r requirements.txt
```

Then run:

```bash
streamlit run app/app.py
```

---

## 👨‍💻 Author

Youssef Samy Youssef