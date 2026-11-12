import streamlit as st
import pandas as pd
import numpy as np
import pickle
from huggingface_hub import hf_hub_download

from pathlib import Path

# PAGE CONFIG
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# LOAD MODEL FROM HUGGING FACE
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="OLASUNKANMI1/house-price-predictor",
        filename="house_price_model.pkl"
    )

    with open(model_path, "rb") as file:
        return pickle.load(file)

model = load_model()

# LOAD DATA
@st.cache_data
def load_data():
    return pd.read_csv("Melbourne_housing_FULL.csv")

df = load_data()

# PREPARE DATE FEATURES
dates = pd.to_datetime(
    df["Date"],
    dayfirst=True,
    errors="coerce"
)

df["SaleYear"] = dates.dt.year
df["SaleMonth"] = dates.dt.month

# CUSTOM CSS
st.markdown("""
<style>
.stApp {
    background: #0b0d12;
    color: white;
}

[data-testid="stSidebar"] {
    background: #151923;
}

h1, h2, h3 {
    color: white;
}

div.stButton > button {
    background: #8b5cf6;
    color: white;
    border-radius: 10px;
    border: none;
    height: 45px;
    font-weight: bold;
}

div.stButton > button:hover {
    background: #7c3aed;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.title("🏠 HOUSE PREDICTOR")
st.sidebar.caption("Melbourne Property Intelligence")

page = st.sidebar.radio(
    "Navigation",
    ["Home", "Price Prediction", "About"]
)

# HOME PAGE
if page == "Home":

    st.title("🏠 Melbourne House Price Predictor")

    st.write(
        "Predict Melbourne property prices using "
        "Machine Learning."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Dataset Records", "34,857")
    col2.metric("ML Algorithm", "Random Forest")
    col3.metric("Test R² Score", "79.08%")

    st.divider()

    st.subheader("Welcome")

    st.write("""
    This application uses a trained Random Forest model
    to estimate Melbourne property prices.

    Enter property information to receive an estimated
    market price.
    """)

    st.info(
        "Navigate to Price Prediction to estimate a property."
    )

# PREDICTION PAGE
elif page == "Price Prediction":

    st.title("Property Price Estimator 🏡")

    st.write("Enter the property information below.")

    with st.form("prediction_form"):

        st.subheader("Location")

        col1, col2, col3 = st.columns(3)

        with col1:
            suburb = st.selectbox(
                "Suburb",
                sorted(df["Suburb"].dropna().unique())
            )

        with col2:
            region = st.selectbox(
                "Region",
                sorted(df["Regionname"].dropna().unique())
            )

        with col3:
            council = st.selectbox(
                "Council Area",
                sorted(df["CouncilArea"].dropna().unique())
            )

        st.divider()

        st.subheader("Property Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            rooms = st.number_input(
                "Rooms", 1, 20, 3
            )

            bedrooms = st.number_input(
                "Bedrooms", 0, 20, 3
            )

            bathrooms = st.number_input(
                "Bathrooms", 0, 15, 2
            )

        with col2:
            property_type = st.selectbox(
                "Property Type",
                ["h", "u", "t"],
                format_func=lambda x: {
                    "h": "House",
                    "u": "Unit / Apartment",
                    "t": "Townhouse"
                }[x]
            )

            car = st.number_input(
                "Car Spaces", 0, 20, 1
            )

            landsize = st.number_input(
                "Land Size (m²)", 0.0, 100000.0, 500.0
            )

        with col3:
            building_area = st.number_input(
                "Building Area (m²)", 0.0, 10000.0, 150.0
            )

            year_built = st.number_input(
                "Year Built", 1800, 2026, 1980
            )

            distance = st.number_input(
                "Distance from CBD (km)",
                0.0, 100.0, 10.0
            )

        st.divider()

        st.subheader("Additional Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            postcode = st.selectbox(
                "Postcode",
                sorted(df["Postcode"].dropna().unique().astype(int))
            )

        with col2:
            propertycount = st.number_input(
                "Property Count", 0, 100000, 7000
            )

        with col3:
            sale_year = st.selectbox(
                "Sale Year", [2016, 2017, 2018], index=1
            )

        sale_month = st.selectbox(
            "Sale Month",
            list(range(1, 13)),
            format_func=lambda x: pd.Timestamp(
                2000, x, 1
            ).strftime("%B")
        )

        st.subheader("Geographical Coordinates")

        col1, col2 = st.columns(2)

        with col1:
            latitude = st.number_input(
                "Latitude", -90.0, 90.0, -37.81
            )

        with col2:
            longitude = st.number_input(
                "Longitude", -180.0, 180.0, 145.00
            )

        submitted = st.form_submit_button(
            "Predict House Price",
            use_container_width=True
        )

    if submitted:

        input_data = pd.DataFrame([{
            "Suburb": suburb,
            "Rooms": rooms,
            "Type": property_type,
            "Distance": distance,
            "Postcode": postcode,
            "Bedroom2": bedrooms,
            "Bathroom": bathrooms,
            "Car": car,
            "Landsize": landsize,
            "BuildingArea": building_area,
            "YearBuilt": year_built,
            "CouncilArea": council,
            "Lattitude": latitude,
            "Longtitude": longitude,
            "Regionname": region,
            "Propertycount": propertycount,
            "SaleYear": sale_year,
            "SaleMonth": sale_month
        }])

        try:
            prediction = model.predict(input_data)[0]

            st.success("Prediction completed!")

            st.metric(
                "Estimated House Price",
                f"${prediction:,.2f}"
            )

            st.write("Property Information")
            st.dataframe(
                input_data,
                use_container_width=True
            )

        except Exception as error:
            st.error(f"Prediction error: {error}")

# ABOUT PAGE
elif page == "About":

    st.title("About This Project")

    st.write("""
    Melbourne House Price Predictor is a machine learning
    application built with Python and Streamlit.

    Machine Learning Algorithm:
    Random Forest Regressor

    Dataset:
    Melbourne Housing Dataset

    Evaluation:
    R² Score: 79.08%

    The model provides estimates for educational purposes
    and is not a professional property valuation.
    """)

# import streamlit as st
# import pandas as pd
# import numpy as np
# import pickle
# from huggingface_hub import hf_hub_download

# from pathlib import Path

# # PAGE CONFIG
# st.set_page_config(
#     page_title="House Price Predictor",
#     page_icon="🏠",
#     layout="wide"
# )

# # LOAD MODEL
# @st.cache_resource
# def load_model():
#     with open("house_price_model.pkl", "rb") as file:
#         return pickle.load(file)

# model = load_model()

# # LOAD DATA
# @st.cache_data
# def load_data():
#     return pd.read_csv("Melbourne_housing_FULL.csv")

# df = load_data()

# # PREPARE DATE FEATURES
# dates = pd.to_datetime(
#     df["Date"],
#     dayfirst=True,
#     errors="coerce"
# )

# df["SaleYear"] = dates.dt.year
# df["SaleMonth"] = dates.dt.month

# # CUSTOM CSS
# st.markdown("""
# <style>
# .stApp {
#     background: #0b0d12;
#     color: white;
# }

# [data-testid="stSidebar"] {
#     background: #151923;
# }

# h1, h2, h3 {
#     color: white;
# }

# div.stButton > button {
#     background: #8b5cf6;
#     color: white;
#     border-radius: 10px;
#     border: none;
#     height: 45px;
#     font-weight: bold;
# }

# div.stButton > button:hover {
#     background: #7c3aed;
#     color: white;
# }
# </style>
# """, unsafe_allow_html=True)

# # SIDEBAR
# st.sidebar.title("🏠 HOUSE PREDICTOR")
# st.sidebar.caption("Melbourne Property Intelligence")

# page = st.sidebar.radio(
#     "Navigation",
#     ["Home", "Price Prediction", "About"]
# )

# # HOME PAGE
# if page == "Home":

#     st.title("🏠 Melbourne House Price Predictor")

#     st.write(
#         "Predict Melbourne property prices using "
#         "Machine Learning."
#     )

#     col1, col2, col3 = st.columns(3)

#     col1.metric("Dataset Records", "34,857")
#     col2.metric("ML Algorithm", "Random Forest")
#     col3.metric("Test R² Score", "79.08%")

#     st.divider()

#     st.subheader("Welcome")

#     st.write("""
#     This application uses a trained Random Forest model
#     to estimate Melbourne property prices.

#     Enter property information to receive an estimated
#     market price.
#     """)

#     st.info(
#         "Navigate to Price Prediction to estimate a property."
#     )

# # PREDICTION PAGE
# elif page == "Price Prediction":

#     st.title("Property Price Estimator 🏡")

#     st.write("Enter the property information below.")

#     with st.form("prediction_form"):

#         st.subheader("Location")

#         col1, col2, col3 = st.columns(3)

#         with col1:
#             suburb = st.selectbox(
#                 "Suburb",
#                 sorted(df["Suburb"].dropna().unique())
#             )

#         with col2:
#             region = st.selectbox(
#                 "Region",
#                 sorted(df["Regionname"].dropna().unique())
#             )

#         with col3:
#             council = st.selectbox(
#                 "Council Area",
#                 sorted(df["CouncilArea"].dropna().unique())
#             )

#         st.divider()

#         st.subheader("Property Details")

#         col1, col2, col3 = st.columns(3)

#         with col1:
#             rooms = st.number_input(
#                 "Rooms", 1, 20, 3
#             )

#             bedrooms = st.number_input(
#                 "Bedrooms", 0, 20, 3
#             )

#             bathrooms = st.number_input(
#                 "Bathrooms", 0, 15, 2
#             )

#         with col2:
#             property_type = st.selectbox(
#                 "Property Type",
#                 ["h", "u", "t"],
#                 format_func=lambda x: {
#                     "h": "House",
#                     "u": "Unit / Apartment",
#                     "t": "Townhouse"
#                 }[x]
#             )

#             car = st.number_input(
#                 "Car Spaces", 0, 20, 1
#             )

#             landsize = st.number_input(
#                 "Land Size (m²)", 0.0, 100000.0, 500.0
#             )

#         with col3:
#             building_area = st.number_input(
#                 "Building Area (m²)", 0.0, 10000.0, 150.0
#             )

#             year_built = st.number_input(
#                 "Year Built", 1800, 2026, 1980
#             )

#             distance = st.number_input(
#                 "Distance from CBD (km)",
#                 0.0, 100.0, 10.0
#             )

#         st.divider()

#         st.subheader("Additional Information")

#         col1, col2, col3 = st.columns(3)

#         with col1:
#             postcode = st.selectbox(
#                 "Postcode",
#                 sorted(df["Postcode"].dropna().unique().astype(int))
#             )

#         with col2:
#             propertycount = st.number_input(
#                 "Property Count", 0, 100000, 7000
#             )

#         with col3:
#             sale_year = st.selectbox(
#                 "Sale Year", [2016, 2017, 2018], index=1
#             )

#         sale_month = st.selectbox(
#             "Sale Month",
#             list(range(1, 13)),
#             format_func=lambda x: pd.Timestamp(
#                 2000, x, 1
#             ).strftime("%B")
#         )

#         st.subheader("Geographical Coordinates")

#         col1, col2 = st.columns(2)

#         with col1:
#             latitude = st.number_input(
#                 "Latitude", -90.0, 90.0, -37.81
#             )

#         with col2:
#             longitude = st.number_input(
#                 "Longitude", -180.0, 180.0, 145.00
#             )

#         submitted = st.form_submit_button(
#             "Predict House Price",
#             use_container_width=True
#         )

#     if submitted:

#         input_data = pd.DataFrame([{
#             "Suburb": suburb,
#             "Rooms": rooms,
#             "Type": property_type,
#             "Distance": distance,
#             "Postcode": postcode,
#             "Bedroom2": bedrooms,
#             "Bathroom": bathrooms,
#             "Car": car,
#             "Landsize": landsize,
#             "BuildingArea": building_area,
#             "YearBuilt": year_built,
#             "CouncilArea": council,
#             "Lattitude": latitude,
#             "Longtitude": longitude,
#             "Regionname": region,
#             "Propertycount": propertycount,
#             "SaleYear": sale_year,
#             "SaleMonth": sale_month
#         }])

#         try:
#             prediction = model.predict(input_data)[0]

#             st.success("Prediction completed!")

#             st.metric(
#                 "Estimated House Price",
#                 f"${prediction:,.2f}"
#             )

#             st.write("Property Information")
#             st.dataframe(
#                 input_data,
#                 use_container_width=True
#             )

#         except Exception as error:
#             st.error(f"Prediction error: {error}")

# # ABOUT PAGE
# elif page == "About":

#     st.title("About This Project")

#     st.write("""
#     Melbourne House Price Predictor is a machine learning
#     application built with Python and Streamlit.

#     Machine Learning Algorithm:
#     Random Forest Regressor

#     Dataset:
#     Melbourne Housing Dataset

#     Evaluation:
#     R² Score: 79.08%
#     the model provides  estimates for educational purposes
    

#     The model provides estimates for educational purposes
#     and is not a professional property valuation.
#     """)
