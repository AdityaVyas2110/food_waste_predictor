import streamlit as st
import pandas as pd
import pickle

st.title("🍽️ Food Waste Reduction Predictor")

# Load model & transformer
model = pickle.load(open("../food_demand_model.pkl","rb"))
transformer = pickle.load(open("../transformer.pkl","rb"))

# ----------------------
# 1️⃣ User inputs for key features
# ----------------------
center_id = st.number_input("Center ID", min_value=1, step=1)
meal_id = st.number_input("Meal ID", min_value=1, step=1)
category = st.selectbox("Category", ["Soup","Starter","Main Course","Dessert"])
cuisine = st.selectbox("Cuisine", ["Indian","Continental","Asian","Mexican"])
center_type = st.selectbox("Center Type", ["Tier 1","Tier 2","Tier 3"])
checkout_price = st.number_input("Checkout Price", min_value=5.0, max_value=50.0, step=0.5)
base_price = st.number_input("Base Price", min_value=5.0, max_value=50.0, step=0.5)
week = st.number_input("Week (1-52)", min_value=1, max_value=52, step=1)

# ----------------------
# 2️⃣ Default values for remaining features
# ----------------------
op_area = 1
city_code = 1
region_code = 1
homepage_featured = 0
emailer_for_promotion = 0
id_col = 99999

# ----------------------
# 3️⃣ Prepare input DataFrame
# ----------------------
sample = pd.DataFrame([{
    "week": week,
    "center_id": center_id,
    "meal_id": meal_id,
    "checkout_price": checkout_price,
    "base_price": base_price,
    "category": category,
    "cuisine": cuisine,
    "center_type": center_type,
    "op_area": op_area,
    "city_code": city_code,
    "region_code": region_code,
    "homepage_featured": homepage_featured,
    "emailer_for_promotion": emailer_for_promotion,
    "id": id_col
}])

# ----------------------
# 4️⃣ Transform & predict
# ----------------------
sample_enc = transformer.transform(sample)
pred = model.predict(sample_enc)

st.success(f"Recommended # of meals to prepare: {int(pred[0])}")
