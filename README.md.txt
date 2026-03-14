# 🍽️ Food Waste Reduction Predictor

A machine learning project that predicts how many meals a restaurant or hostel should prepare each day. The goal is to **reduce food waste, save costs, and improve kitchen efficiency**.

---

## 1. Problem Statement

Restaurants and hostels often **overcook or undercook meals** because they cannot accurately predict daily demand.  
This leads to:

- Food being wasted  
- Higher costs for ingredients and labor  
- Inefficient kitchen operations  

This project provides a **data-driven solution** to predict the number of meals required each day.

---

## 2. How the Project Works

1. **Collect Data:**  
   - Historical meal orders per center  
   - Meal information (type, category, cuisine)  
   - Center information (location, tier)

2. **Train Machine Learning Model:**  
   - Uses historical patterns to learn the relationship between features (meal, center, week, price) and number of orders.  
   - RandomForestRegressor is used for prediction.  

3. **Make Predictions:**  
   - Input meal details, center, week, and prices.  
   - Model predicts the **number of meals to prepare**.  

4. **Interactive Dashboard:**  
   - Streamlit dashboard allows managers to input parameters and get **instant predictions**.

---

## 3. Dataset

We use the **Kaggle Food Demand Forecasting Dataset**, which contains:

| File | Description |
|------|-------------|
| `train.csv` | Historical orders for each meal at each center |
| `meal_info.csv` | Meal metadata (category, cuisine) |
| `fulfilment_center_info.csv` | Center metadata (location, tier) |

**Key Features Used:**

- `meal_id`, `center_id`  
- `checkout_price`, `base_price`  
- `category`, `cuisine`, `center_type`  
- `week`, `op_area`, `city_code`, `region_code`  
- `homepage_featured`, `emailer_for_promotion`  
- `id` (unique row identifier)

---

## 4. Project Structure
food_waste_predictor/
│
├── data/
│ ├── train.csv
│ ├── meal_info.csv
│ └── fulfilment_center_info.csv
│
├── src/
│ ├── train_model.py # Script to train the ML model
│ ├── predict.py # Command-line prediction example
│ └── dashboard.py # Streamlit dashboard
│
├── requirements.txt
└── README.md


---

## 5. Installation

1. Clone the repository:

git clone <repository_url>
cd food_waste_predictor

2. Install Python dependencies:

pip install -r requirements.txt

## 6. Usage

6.1 Train the Model

python src/train_model.py
Trains a RandomForestRegressor

Saves the model as food_demand_model.pkl

Saves the transformer as transformer.pkl


6.2 Make a Prediction (Command-Line)
python src/predict.py

Predicts the number of meals for a single input

Modify the sample dictionary in the script to test different meals, centers, or weeks

6.3 Run Streamlit Dashboard
streamlit run src/dashboard.py

Enter meal details, center info, week, and prices

Get recommended number of meals to prepare instant

## 7. Real-World Impact

Reduces food waste: Only cook what is needed

Saves money: Fewer wasted ingredients

Optimizes kitchen operations: Staff can plan meals accurately

Scalable: Can be applied to multiple restaurants or hostels

## 8. Challenges Solved

High-cardinality categorical features → handled using LabelEncoder

Transformer requires all columns → default values filled for missing columns

Large dataset → sampled for faster training during development

Predictions were initially constant → varying input features fixed this

9. Future Improvements

Add holiday and weather data for more accurate predictions

Predict weekly trends instead of single-day demand

Visualize waste reduction and predicted demand trends on dashboard

Automatically update the model with new data daily

10. Key Takeaways

This project demonstrates how machine learning can solve a real-world operational problem:

Uses historical data to make actionable predictions

Helps businesses reduce waste and save money

Provides a user-friendly dashboard for real-time decision making