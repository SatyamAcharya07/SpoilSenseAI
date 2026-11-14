
import pickle
from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Food Spoilage Predictor",
    page_icon="🥬",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pkl"
DATA_PATH = BASE_DIR / "data" / "raw" / "data.csv"


# =====================================================
# CUSTOM STYLING
# =====================================================

st.markdown("""
<style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    div[data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.08);
        padding: 15px;
        border-radius: 10px;
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
    }

    .subtitle {
        color: gray;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# =====================================================
# LOAD MODEL AND DATASET
# =====================================================

@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        with open(MODEL_PATH, "rb") as file:
            return pickle.load(file)


@st.cache_data
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    df = df.drop_duplicates().copy()

    df["Dispatch_Date"] = pd.to_datetime(
        df["Dispatch_Date"], errors="coerce"
    )

    return df


if not MODEL_PATH.exists():
    st.error("model.pkl not found. Place it in the project root.")
    st.stop()

if not DATA_PATH.exists():
    st.error("Dataset not found at data/raw/data.csv.")
    st.stop()

try:
    model = load_model()
    df = load_dataset()
except Exception as error:
    st.error("Unable to load the model or dataset.")
    st.exception(error)
    st.stop()


# =====================================================
# SIDEBAR NAVIGATION
# =====================================================

st.sidebar.title("🥬 FoodSpoil AI")
st.sidebar.caption("Food Supply Chain Intelligence")
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Predict Shelf Life",
        "Data Analysis",
        "Model Information"
    ]
)

st.sidebar.divider()
st.sidebar.caption("Machine Learning Project | Ridge Regression")


# =====================================================
# DASHBOARD PAGE
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<p class="main-title">Food Supply Chain Dashboard</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="subtitle">Monitor shipment data and understand '
        'food shelf-life patterns using machine learning.</p>',
        unsafe_allow_html=True
    )

    st.divider()

    total_shipments = len(df)
    total_products = df["Product_Type"].nunique()
    avg_shelf_life = df["Remaining_Shelf_Life_Hours"].mean()
    avg_temperature = df["Avg_Temperature_C"].mean()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Shipments", f"{total_shipments:,}")
    col2.metric("Product Categories", total_products)
    col3.metric("Average Shelf Life", f"{avg_shelf_life:.1f} hrs")
    col4.metric("Average Temperature", f"{avg_temperature:.1f} °C")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Shipments by Product Type")

        product_counts = (
            df["Product_Type"]
            .value_counts()
            .reset_index()
        )
        product_counts.columns = ["Product Type", "Shipments"]

        fig = px.bar(
            product_counts,
            x="Product Type",
            y="Shipments",
            color="Product Type",
            title="Distribution of Product Categories"
        )

        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Shelf Life Distribution")

        fig = px.histogram(
            df,
            x="Remaining_Shelf_Life_Hours",
            nbins=30,
            title="Remaining Shelf Life Across Shipments",
            labels={
                "Remaining_Shelf_Life_Hours": "Shelf Life (Hours)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Shipment Records")

    display_columns = [
        "Shipment_ID",
        "Product_Type",
        "Storage_Condition",
        "Avg_Temperature_C",
        "Relative_Humidity_pct",
        "Remaining_Shelf_Life_Hours"
    ]

    st.dataframe(
        df[display_columns].head(10),
        use_container_width=True,
        hide_index=True
    )


# =====================================================
# PREDICTION PAGE
# =====================================================

elif page == "Predict Shelf Life":

    st.title("🔍 Remaining Shelf Life Prediction")

    st.write(
        "Enter the shipment details to estimate the remaining "
        "shelf life of your food product."
    )

    st.info(
        "Provide the available shipment measurements. "
        "The model will estimate the remaining shelf life in hours."
    )

    with st.form("prediction_form"):

        st.subheader("1. Product Information")

        col1, col2 = st.columns(2)

        with col1:
            product_type = st.selectbox(
                "Product Type",
                [
                    "Leafy Greens",
                    "Tomatoes",
                    "Strawberries",
                    "Blueberries",
                    "Bell Peppers",
                    "Bananas"
                ]
            )

            storage_condition = st.selectbox(
                "Storage Condition",
                [
                    "Refrigerated",
                    "Insulated",
                    "Ambient"
                ]
            )

            initial_shelf_life = st.number_input(
                "Initial Shelf Life (Hours)",
                min_value=0.0,
                max_value=500.0,
                value=120.0
            )

        with col2:
            dispatch_month = st.selectbox(
                "Dispatch Month",
                list(range(1, 13)),
                format_func=lambda x: pd.Timestamp(
                    year=2025, month=x, day=1
                ).strftime("%B")
            )

        st.divider()

        st.subheader("2. Temperature and Humidity")

        col1, col2, col3 = st.columns(3)

        with col1:
            avg_temperature = st.number_input(
                "Average Temperature (°C)",
                min_value=-10.0,
                max_value=50.0,
                value=8.0
            )

            min_temperature = st.number_input(
                "Minimum Temperature (°C)",
                min_value=-20.0,
                max_value=50.0,
                value=4.0
            )

        with col2:
            max_temperature = st.number_input(
                "Maximum Temperature (°C)",
                min_value=-10.0,
                max_value=60.0,
                value=12.0
            )

            temperature_fluctuation = st.number_input(
                "Temperature Fluctuation (°C)",
                min_value=0.0,
                max_value=30.0,
                value=5.0
            )

        with col3:
            humidity = st.number_input(
                "Relative Humidity (%)",
                min_value=0.0,
                max_value=100.0,
                value=90.0
            )

        st.divider()

        st.subheader("3. Transportation Information")

        col1, col2 = st.columns(2)

        with col1:
            transit_elapsed = st.number_input(
                "Transit Elapsed (Hours)",
                min_value=0.0,
                max_value=200.0,
                value=24.0
            )

            expected_transit_remaining = st.number_input(
                "Expected Transit Remaining (Hours)",
                min_value=0.0,
                max_value=200.0,
                value=24.0
            )

        with col2:
            hours_above_threshold = st.number_input(
                "Hours Above Product Temperature Threshold",
                min_value=0.0,
                max_value=200.0,
                value=2.0
            )

            unplanned_delay = st.number_input(
                "Unplanned Delay (Hours)",
                min_value=0.0,
                max_value=100.0,
                value=1.0
            )

        st.divider()

        submitted = st.form_submit_button(
            "Predict Remaining Shelf Life",
            use_container_width=True,
            type="primary"
        )

    if submitted:

        if not (
            min_temperature <= avg_temperature <= max_temperature
        ):
            st.error(
                "Invalid temperature values. Ensure Minimum ≤ Average ≤ Maximum."
            )

        else:

            input_data = pd.DataFrame([{
                "Product_Type": product_type,
                "Storage_Condition": storage_condition,
                "Initial_Shelf_Life_Hours": initial_shelf_life,
                "Avg_Temperature_C": avg_temperature,
                "Min_Temperature_C": min_temperature,
                "Max_Temperature_C": max_temperature,
                "Temperature_Fluctuation_C": temperature_fluctuation,
                "Relative_Humidity_pct": humidity,
                "Transit_Elapsed_Hours": transit_elapsed,
                "Expected_Transit_Remaining_Hours": expected_transit_remaining,
                "Hours_Above_Product_Temp_Threshold": hours_above_threshold,
                "Unplanned_Delay_Hours": unplanned_delay,
                "Dispatch_Month": dispatch_month
            }])

            try:
                prediction = float(model.predict(input_data)[0])

                st.divider()
                st.subheader("Prediction Result")

                col1, col2 = st.columns(2)

                col1.metric(
                    "Estimated Remaining Shelf Life",
                    f"{prediction:.2f} Hours"
                )

                col2.metric(
                    "Approximate Remaining Days",
                    f"{prediction / 24:.2f} Days"
                )

                st.success("Prediction generated successfully.")

                st.warning(
                    "This is an experimental ML estimate based on synthetic "
                    "training data. It is not a food-safety assessment."
                )

            except Exception as error:
                st.error("Prediction failed.")
                st.exception(error)


# =====================================================
# DATA ANALYSIS PAGE
# =====================================================

elif page == "Data Analysis":

    st.title("📊 Exploratory Data Analysis")

    st.write(
        "Explore the dataset and understand relationships "
        "between shipment conditions and remaining shelf life."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "Target Distribution",
        "Feature Relationships",
        "Correlation",
        "Product Comparison"
    ])

    with tab1:

        st.subheader("Remaining Shelf Life Distribution")

        fig = px.histogram(
            df,
            x="Remaining_Shelf_Life_Hours",
            nbins=30,
            marginal="box",
            title="Distribution of Remaining Shelf Life",
            labels={
                "Remaining_Shelf_Life_Hours": "Remaining Shelf Life (Hours)"
            }
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:

        st.subheader("Feature vs Remaining Shelf Life")

        feature = st.selectbox(
            "Select a feature",
            [
                "Avg_Temperature_C",
                "Relative_Humidity_pct",
                "Transit_Elapsed_Hours",
                "Unplanned_Delay_Hours",
                "Hours_Above_Product_Temp_Threshold",
                "Initial_Shelf_Life_Hours"
            ]
        )

        fig = px.scatter(
            df,
            x=feature,
            y="Remaining_Shelf_Life_Hours",
            color="Product_Type",
            opacity=0.65,
            title=f"{feature} vs Remaining Shelf Life"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:

        st.subheader("Correlation Heatmap")

        numerical_df = df.select_dtypes(include="number")

        correlation = numerical_df.corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="RdBu_r",
            zmin=-1,
            zmax=1,
            aspect="auto",
            title="Numerical Feature Correlation"
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab4:

        st.subheader("Remaining Shelf Life by Product Type")

        fig = px.box(
            df,
            x="Product_Type",
            y="Remaining_Shelf_Life_Hours",
            color="Product_Type",
            title="Shelf Life Comparison Across Products"
        )

        st.plotly_chart(fig, use_container_width=True)


# =====================================================
# MODEL INFORMATION PAGE
# =====================================================

elif page == "Model Information":

    st.title("🤖 Model Information")

    st.write(
        "This application uses a trained Ridge Regression pipeline "
        "to estimate the remaining shelf life of food shipments."
    )

    st.subheader("Selected Algorithm")

    st.markdown("### Ridge Regression")

    st.write(
        "Ridge Regression is a regularised linear regression algorithm. "
        "It uses L2 regularisation to help control model coefficients."
    )

    st.divider()

    st.subheader("Model Evaluation")

    col1, col2, col3 = st.columns(3)

    col1.metric("MAE", "7.71 Hours")
    col2.metric("RMSE", "14.85 Hours")
    col3.metric("R² Score", "0.9323")

    st.caption(
        "Metrics are from the notebook's reported test-set evaluation."
    )

    st.divider()

    st.subheader("Model Input Features")

    features = [
        "Product_Type",
        "Storage_Condition",
        "Initial_Shelf_Life_Hours",
        "Avg_Temperature_C",
        "Min_Temperature_C",
        "Max_Temperature_C",
        "Temperature_Fluctuation_C",
        "Relative_Humidity_pct",
        "Transit_Elapsed_Hours",
        "Expected_Transit_Remaining_Hours",
        "Hours_Above_Product_Temp_Threshold",
        "Unplanned_Delay_Hours",
        "Dispatch_Month"
    ]

    st.write(features)

    st.divider()

    st.subheader("Project Limitations")

    st.warning(
        "The model was trained on a synthetic dataset. Its performance "
        "has not been validated using real-world food shipment data. "
        "Predictions should be treated as experimental estimates, "
        "not food quality or safety guarantees."
    )

    st.caption(
        "Developed using Python, Pandas, Scikit-learn, Plotly and Streamlit."
    )
