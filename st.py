import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Set page config first
st.set_page_config(layout="wide")

# Load the dataset
@st.cache_data
def load_data():
    data = pd.read_csv("E:\\mov_rev\\Housing.csv")
    return data

data = load_data()

# Encode categorical variables
label_encoder = LabelEncoder()
data['mainroad'] = label_encoder.fit_transform(data['mainroad'])
data['guestroom'] = label_encoder.fit_transform(data['guestroom'])
data['basement'] = label_encoder.fit_transform(data['basement'])
data['hotwaterheating'] = label_encoder.fit_transform(data['hotwaterheating'])
data['airconditioning'] = label_encoder.fit_transform(data['airconditioning'])
data['prefarea'] = label_encoder.fit_transform(data['prefarea'])
data['furnishingstatus'] = label_encoder.fit_transform(data['furnishingstatus'])

# Split the data into features and target
X = data.drop("price", axis=1)
y = data["price"]

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Create an XGBoost model with manually set hyperparameters
model = xgb.XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate the metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# Title and Image
st.title("House Price Prediction App")
st.image("https://media.istockphoto.com/id/1415886888/photo/freshly-painted-craftsman-bungalow-house.jpg?s=612x612&w=0&k=20&c=uzf_2Zl4MPpvE8J8PzJeJLaXqyyXpOP1YvWsUbpns5g=", width=100)

# Sidebar Metrics
st.sidebar.header("Model Metrics")
st.sidebar.write("Mean Squared Error:", mse)
st.sidebar.write("Mean Absolute Error:", mae)

# Sidebar Plot
st.sidebar.header("Actual vs. Predicted Prices")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', linewidth=2)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs. Predicted Prices")
st.sidebar.pyplot(plt)

# Input features for prediction
st.header("Input Features")
def user_input_features():
    area = st.number_input("Area (sq. ft.)", value=7420)
    bedrooms = st.number_input("Bedrooms", value=4)
    bathrooms = st.number_input("Bathrooms", value=2)
    stories = st.number_input("Stories", value=3)
    mainroad = st.selectbox("Main Road Access", ("yes", "no"))
    guestroom = st.selectbox("Guest Room", ("yes", "no"))
    basement = st.selectbox("Basement", ("yes", "no"))
    hotwaterheating = st.selectbox("Hot Water Heating", ("yes", "no"))
    airconditioning = st.selectbox("Air Conditioning", ("yes", "no"))
    parking = st.number_input("Parking Spaces", value=2)
    prefarea = st.selectbox("Preferred Area", ("yes", "no"))
    furnishingstatus = st.selectbox("Furnishing Status", ("furnished", "semi-furnished", "unfurnished"))

    input_data = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "mainroad": 1 if mainroad == "yes" else 0,
        "guestroom": 1 if guestroom == "yes" else 0,
        "basement": 1 if basement == "yes" else 0,
        "hotwaterheating": 1 if hotwaterheating == "yes" else 0,
        "airconditioning": 1 if airconditioning == "yes" else 0,
        "parking": parking,
        "prefarea": 1 if prefarea == "yes" else 0,
        "furnishingstatus": label_encoder.transform([furnishingstatus])[0]
    }
    features = pd.DataFrame(input_data, index=[0])
    return features

input_df = user_input_features()

# Predict the price
if st.button("Predict Price"):
    input_df_scaled = scaler.transform(input_df)
    prediction = model.predict(input_df_scaled)
    st.write("Predicted House Price: ₹", prediction[0])
