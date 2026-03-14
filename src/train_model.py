import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor

# ----------------------
# 1️⃣ Load datasets
# ----------------------
train = pd.read_csv(r"C:\Users\ADITYA VYAS\food_waste_predictor\data\train.csv")
meals = pd.read_csv(r"C:\Users\ADITYA VYAS\food_waste_predictor\data\meal_info.csv")
centers = pd.read_csv(r"C:\Users\ADITYA VYAS\food_waste_predictor\data\fulfilment_center_info.csv")

# Merge datasets
df = train.merge(meals, on="meal_id", how="left")
df = df.merge(centers, on="center_id", how="left")

# Optional: Use smaller sample for testing speed
df = df.sample(10000, random_state=42)

# ----------------------
# 2️⃣ Prepare features
# ----------------------
# Encode high-cardinality categorical features
for col in ["meal_id","center_id"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Target and features
y = df["num_orders"]
X = df.drop("num_orders", axis=1)

# Identify columns
cat_cols = ["category","cuisine","center_type"]
num_cols = ["checkout_price","base_price","week","op_area","city_code","region_code",
            "homepage_featured","emailer_for_promotion","id"]

# ColumnTransformer
ct = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
], remainder="passthrough")

X_enc = ct.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_enc, y, test_size=0.2, random_state=42)

# ----------------------
# 3️⃣ Train model
# ----------------------
model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Evaluate
preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
print(f"RandomForest MAE: {mae}")

# ----------------------
# 4️⃣ Save model and transformer
# ----------------------
pickle.dump(model, open("../food_demand_model.pkl","wb"))
pickle.dump(ct, open("../transformer.pkl","wb"))
print("Model and transformer saved successfully!")
