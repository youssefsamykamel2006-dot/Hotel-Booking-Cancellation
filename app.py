# ==================================================
# Hotel Booking Cancellation Dashboard
# ==================================================

# ==================================================
# 1. Import Libraries
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

import joblib

from streamlit_option_menu import option_menu


# ==================================================
# 2. Page Configuration
# ==================================================

st.set_page_config(

    page_title="Hotel Booking Cancellation Dashboard",

    page_icon="🏨",

    layout="wide",

    initial_sidebar_state="expanded"

)


# ==================================================
# 3. Load Dataset
# ==================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "D:/datascience222/youssef samy youssef (final project)/Hotel-Booking-Cancellation/data/hotel_bookings_final.csv"
    )

    return df


# ==================================================
# 4. Load Model
# ==================================================

@st.cache_resource
def load_model():

    model = joblib.load(
        "D:/datascience222/youssef samy youssef (final project)/Hotel-Booking-Cancellation/models/best_xgboost_model.pkl"
    )

    return model


df = load_data()

model = load_model()


# ==================================================
# 5. Sidebar
# ==================================================

with st.sidebar:

    st.title("🏨 Hotel Booking")

    st.caption("Cancellation Prediction Dashboard")

    selected = option_menu(

        menu_title=None,

        options=[
            "Dashboard",
            "Analysis",
            "Prediction",
            "About"
        ],

        icons=[
            "house-fill",
            "bar-chart-fill",
            "robot",
            "info-circle-fill"
        ],

        default_index=0,

        styles={

            "container": {
                "padding": "5!important",
                "background-color": "#fafafa"
            },

            "icon": {
                "color": "#0d6efd",
                "font-size": "18px"
            },

            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "2px"
            },

            "nav-link-selected": {
                "background-color": "#0d6efd"
            }

        }

    )

    st.markdown("---")

    st.caption("Machine Learning Deployment")
# ==================================================
# 6. Dashboard
# ==================================================

if selected == "Dashboard":

    st.title("🏨 Hotel Booking Cancellation Dashboard")

    st.markdown(
        """
        Welcome to the Hotel Booking Cancellation Dashboard.
        Explore hotel booking trends, analyze customer behavior,
        and predict booking cancellations using Machine Learning.
        """
    )

    st.markdown("---")

    # ==========================
    # KPI Cards
    # ==========================

    total_bookings = len(df)

    cancellation_rate = df["is_canceled"].mean() * 100

    average_adr = df["adr"].mean()

    average_guests = df["total_guests"].mean()

    average_nights = df["total_nights"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Bookings", f"{total_bookings:,}")

    c2.metric("Cancellation Rate", f"{cancellation_rate:.2f}%")

    c3.metric("Average ADR", f"${average_adr:.2f}")

    c4.metric("Average Guests", f"{average_guests:.2f}")

    c5.metric("Average Nights", f"{average_nights:.2f}")

    st.markdown("---")

    st.subheader("📌 Key Insights")

    col1, col2 = st.columns(2)

    with col1:

        st.info("""
- City Hotel receives the highest number of bookings.
- Summer months have the highest demand.
- Long lead time increases cancellation probability.
        """)

    with col2:

        st.success("""
- Non-refundable deposits reduce cancellations.
- Repeated guests rarely cancel.
- More special requests indicate committed guests.
        """)

    st.markdown("---")

    # ==========================
    # Charts Row 1
    # ==========================

    col1, col2 = st.columns(2)

    with col1:

        hotel_counts = (
            df["hotel"]
            .value_counts()
            .reset_index()
        )

        hotel_counts.columns = ["Hotel", "Bookings"]

        fig = px.pie(

            hotel_counts,

            names="Hotel",

            values="Bookings",

            hole=0.45,

            title="Hotel Type Distribution"

        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        cancel = (
            df["is_canceled"]
            .value_counts()
            .reset_index()
        )

        cancel.columns = ["Status", "Bookings"]

        cancel["Status"] = cancel["Status"].map({

            0: "Not Cancelled",

            1: "Cancelled"

        })

        fig = px.bar(

            cancel,

            x="Status",

            y="Bookings",

            color="Status",

            text_auto=True,

            title="Cancellation Distribution"

        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ==========================
    # Charts Row 2
    # ==========================

    col1, col2 = st.columns(2)

    with col1:

        monthly = (
            df.groupby("arrival_date_month")
            .size()
            .reset_index(name="Bookings")
        )

        month_order = [

            "January","February","March","April",

            "May","June","July","August",

            "September","October","November","December"

        ]

        monthly["arrival_date_month"] = pd.Categorical(

            monthly["arrival_date_month"],

            categories=month_order,

            ordered=True

        )

        monthly = monthly.sort_values("arrival_date_month")

        fig = px.line(

            monthly,

            x="arrival_date_month",

            y="Bookings",

            markers=True,

            title="Monthly Bookings"

        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(

            df,

            x="lead_time",

            nbins=40,

            title="Lead Time Distribution"

        )

        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 7. Analysis
# ==================================================

if selected == "Analysis":

    st.title("📊 Hotel Booking Analysis")

    st.markdown(
        "Analyze booking trends using interactive filters."
    )

    st.markdown("---")

    # ======================================
    # Filters
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        selected_hotel = st.multiselect(

            "Hotel Type",

            sorted(df["hotel"].unique()),

            default=sorted(df["hotel"].unique())

        )

    with col2:

        selected_market = st.multiselect(

            "Market Segment",

            sorted(df["market_segment"].unique()),

            default=sorted(df["market_segment"].unique())

        )

    filtered_df = df[

        (df["hotel"].isin(selected_hotel))

        &

        (df["market_segment"].isin(selected_market))

    ]

    st.markdown("---")

    # ======================================
    # Charts Row 1
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        cancel = (

            filtered_df["is_canceled"]

            .value_counts()

            .reset_index()

        )

        cancel.columns = ["Status", "Bookings"]

        cancel["Status"] = cancel["Status"].map({

            0: "Not Cancelled",

            1: "Cancelled"

        })

        fig = px.pie(

            cancel,

            names="Status",

            values="Bookings",

            hole=0.45,

            title="Cancellation Distribution"

        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        hotel_cancel = (

            filtered_df

            .groupby("hotel")["is_canceled"]

            .mean()

            .reset_index()

        )

        hotel_cancel["is_canceled"] *= 100

        fig = px.bar(

            hotel_cancel,

            x="hotel",

            y="is_canceled",

            color="hotel",

            text_auto=".1f",

            title="Cancellation Rate by Hotel"

        )

        fig.update_layout(

            yaxis_title="Cancellation Rate (%)"

        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================
    # Charts Row 2
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        fig = px.box(

            filtered_df,

            x="hotel",

            y="adr",

            color="hotel",

            title="ADR by Hotel"

        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        fig = px.histogram(

            filtered_df,

            x="lead_time",

            color="hotel",

            nbins=40,

            title="Lead Time Distribution"

        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================
    # Charts Row 3
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        market = (

            filtered_df["market_segment"]

            .value_counts()

            .reset_index()

        )

        market.columns = [

            "Market Segment",

            "Bookings"

        ]

        fig = px.bar(

            market,

            x="Market Segment",

            y="Bookings",

            color="Market Segment",

            title="Bookings by Market Segment"

        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        deposit = (

            filtered_df

            .groupby("deposit_type")["is_canceled"]

            .mean()

            .reset_index()

        )

        deposit["is_canceled"] *= 100

        fig = px.bar(

            deposit,

            x="deposit_type",

            y="is_canceled",

            color="deposit_type",

            text_auto=".1f",

            title="Cancellation Rate by Deposit Type"

        )

        fig.update_layout(

            yaxis_title="Cancellation Rate (%)"

        )

        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ======================================
    # Correlation Heatmap
    # ======================================

    st.subheader("Correlation Heatmap")

    corr = filtered_df.select_dtypes(include=np.number).corr()

    fig = px.imshow(

        corr,

        text_auto=".2f",

        color_continuous_scale="RdBu_r",

        title="Correlation Matrix"

    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# 8. Prediction
# ==================================================

if selected == "Prediction":

    st.title("🤖 Booking Cancellation Prediction")

    st.markdown(
        "Fill in the booking information below and click Predict."
    )

    st.markdown("---")

    # ======================================
    # Booking Information
    # ======================================

    col1, col2 = st.columns(2)

    with col1:

        hotel = st.selectbox(
            "Hotel",
            sorted(df["hotel"].unique())
        )

        lead_time = st.number_input(
            "Lead Time",
            min_value=0,
            value=30
        )

        arrival_year = st.selectbox(
            "Arrival Year",
            sorted(df["arrival_date_year"].unique())
        )

        arrival_month = st.selectbox(
            "Arrival Month",
            sorted(df["arrival_date_month"].unique())
        )

        arrival_week = st.slider(
            "Arrival Week Number",
            1,
            53,
            25
        )

        arrival_day = st.slider(
            "Arrival Day",
            1,
            31,
            15
        )

        weekend_nights = st.slider(
            "Weekend Nights",
            0,
            10,
            1
        )

        week_nights = st.slider(
            "Week Nights",
            0,
            20,
            2
        )

        adults = st.number_input(
            "Adults",
            1,
            10,
            2
        )

        children = st.number_input(
            "Children",
            0,
            10,
            0
        )

        babies = st.number_input(
            "Babies",
            0,
            5,
            0
        )

    with col2:

        meal = st.selectbox(
            "Meal",
            sorted(df["meal"].unique())
        )

        country = st.selectbox(
            "Country",
            sorted(df["country"].dropna().unique())
        )

        market_segment = st.selectbox(
            "Market Segment",
            sorted(df["market_segment"].unique())
        )

        distribution_channel = st.selectbox(
            "Distribution Channel",
            sorted(df["distribution_channel"].unique())
        )

        customer_type = st.selectbox(
            "Customer Type",
            sorted(df["customer_type"].unique())
        )

        deposit_type = st.selectbox(
            "Deposit Type",
            sorted(df["deposit_type"].unique())
        )

        reserved_room = st.selectbox(
            "Reserved Room Type",
            sorted(df["reserved_room_type"].unique())
        )

        assigned_room = st.selectbox(
            "Assigned Room Type",
            sorted(df["assigned_room_type"].unique())
        )

        booking_changes = st.number_input(
            "Booking Changes",
            0,
            20,
            0
        )

        previous_cancel = st.number_input(
            "Previous Cancellations",
            0,
            20,
            0
        )

        previous_not_cancel = st.number_input(
            "Previous Bookings Not Cancelled",
            0,
            100,
            0
        )

        parking = st.slider(
            "Parking Spaces",
            0,
            5,
            0
        )

        special_requests = st.slider(
            "Special Requests",
            0,
            5,
            0
        )

        repeated_guest = st.selectbox(
            "Repeated Guest",
            [0, 1]
        )

        adr = st.number_input(
            "Average Daily Rate",
            min_value=0.0,
            value=100.0
        )

    st.markdown("---")

    if st.button("🔮 Predict Cancellation", use_container_width=True):

        total_nights = weekend_nights + week_nights

        total_guests = adults + children + babies

        is_family = 1 if (children + babies) > 0 else 0

        adr_per_guest = adr / total_guests if total_guests > 0 else adr

        input_data = pd.DataFrame({

            "hotel":[hotel],
            "lead_time":[lead_time],
            "arrival_date_year":[arrival_year],
            "arrival_date_month":[arrival_month],
            "arrival_date_week_number":[arrival_week],
            "arrival_date_day_of_month":[arrival_day],
            "stays_in_weekend_nights":[weekend_nights],
            "stays_in_week_nights":[week_nights],
            "adults":[adults],
            "children":[children],
            "babies":[babies],
            "meal":[meal],
            "country":[country],
            "market_segment":[market_segment],
            "distribution_channel":[distribution_channel],
            "is_repeated_guest":[repeated_guest],
            "previous_cancellations":[previous_cancel],
            "previous_bookings_not_canceled":[previous_not_cancel],
            "reserved_room_type":[reserved_room],
            "assigned_room_type":[assigned_room],
            "booking_changes":[booking_changes],
            "deposit_type":[deposit_type],
            "agent":[0],
            "days_in_waiting_list":[0],
            "customer_type":[customer_type],
            "adr":[adr],
            "required_car_parking_spaces":[parking],
            "total_of_special_requests":[special_requests],
            "total_nights":[total_nights],
            "total_guests":[total_guests],
            "is_family":[is_family],
            "adr_per_guest":[adr_per_guest]

        })

        prediction = model.predict(input_data)[0]

        probability = model.predict_proba(input_data)[0]

        st.markdown("---")

        st.subheader("Prediction Result")

        confidence = np.max(probability) * 100

        if prediction == 1:

            st.error("❌ This booking is likely to be Cancelled")

            st.progress(float(probability[1]))

        else:

            st.success("✅ This booking is likely to be Confirmed")

            st.progress(float(probability[0]))

        st.metric(
            "Model Confidence",
            f"{confidence:.2f}%"
        )
    
# ==================================================
# 9. About
# ==================================================

if selected == "About":

    st.title("ℹ️ About This Project")

    st.markdown("---")

    st.subheader("🏨 Project Overview")

    st.write(
        """
        The Hotel Booking Cancellation Dashboard is an interactive
        Machine Learning application designed to analyze hotel booking
        behavior and predict whether a reservation is likely to be
        cancelled.

        The application combines data visualization, business insights,
        and predictive analytics to support better decision-making.
        """
    )

    st.markdown("---")

    st.subheader("📂 Dataset Information")

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "Number of Records",
            f"{len(df):,}"
        )

        st.metric(
            "Number of Features",
            df.shape[1]
        )

    with col2:

        st.metric(
            "Cancellation Rate",
            f"{df['is_canceled'].mean()*100:.2f}%"
        )

        st.metric(
            "Average ADR",
            f"${df['adr'].mean():.2f}"
        )

    st.markdown("---")

    st.subheader("🤖 Machine Learning Model")

    st.write("""
    **Selected Model:** XGBoost Classifier

    The model was selected after comparing multiple machine learning
    algorithms including:

    • Logistic Regression

    • K-Nearest Neighbors (KNN)

    • Decision Tree

    • Random Forest

    • XGBoost

    Hyperparameter tuning was performed using GridSearchCV,
    and the final XGBoost model achieved the best overall performance.
    """)

    st.markdown("---")

    st.subheader("📈 Final Model Performance")

    performance = pd.DataFrame({

        "Metric": [

            "Accuracy",

            "Precision",

            "Recall",

            "F1 Score"

        ],

        "Value": [

            "85.60%",

            "74.82%",

            "71.91%",

            "73.34%"

        ]

    })

    st.dataframe(

        performance,

        use_container_width=True,

        hide_index=True

    )

    st.markdown("---")

    st.subheader("🛠️ Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Plotly
    - Scikit-learn
    - XGBoost
    - Imbalanced-learn (SMOTE)
    - Joblib
    """)

    st.markdown("---")

    st.success(
        "Hotel Booking Cancellation Dashboard | Final Graduation Project"
    )


