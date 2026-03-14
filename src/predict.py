import pandas as pd
import pickle

# Load trained model & transformer
model = pickle.load(open("../food_demand_model.pkl","rb"))
transformer = pickle.load(open("../transformer.pkl","rb"))

# Example input (all columns present)
sample = pd.DataFrame([{
    "week": 12,
    "center_id": 1,
    "meal_id": 10,
    "checkout_price": 12.5,
    "base_price": 10.0,
    "category": "Soup",
    "cuisine": "Indian",
    "center_type": "Tier 1",
    "op_area": 1,
    "city_code": 1,
    "region_code": 1,
    "homepage_featured": 0,
    "emailer_for_promotion": 0,
    "id": 1001
}])

# Transform & predict
sample_enc = transformer.transform(sample)
pred = model.predict(sample_enc)
print("Predicted # of meals:", int(pred[0]))
