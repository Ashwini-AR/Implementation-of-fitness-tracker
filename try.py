import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Page Configuration
st.set_page_config(page_title="Personal Fitness Tracker", layout="wide")
st.title("🏋️ Personal Fitness Tracker")
st.write("### Predict Your Calories Burned Based on Your Personal Data")

# Sidebar - User Input
st.sidebar.header("User Input Parameters")
def user_input_features():
    age = st.sidebar.slider("Age", 10, 100, 30)
    bmi = st.sidebar.slider("BMI", 15, 40, 22)
    duration = st.sidebar.slider("Duration (min)", 1, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate", 60, 180, 90)
    body_temp = st.sidebar.slider("Body Temperature (°C)", 35, 42, 37)
    gender = st.sidebar.radio("Gender", ["Male", "Female"]) == "Male"
    return pd.DataFrame({
        "Age": [age], "BMI": [bmi], "Duration": [duration],
        "Heart_Rate": [heart_rate], "Body_Temp": [body_temp], "Gender_male": [int(gender)]
    })

df = user_input_features()
st.write("#### Your Parameters:")
st.dataframe(df, use_container_width=True)

# Caching Data Loading
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    df = exercise.merge(calories, on="User_ID").drop(columns=["User_ID"])
    df["BMI"] = round(df["Weight"] / (df["Height"] / 100) ** 2, 2)
    df = df[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    df = pd.get_dummies(df, drop_first=True)
    return df

data = load_data()

# Split Data
X = data.drop("Calories", axis=1)
y = data["Calories"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Selection
st.sidebar.subheader("Choose Model")
model_choice = st.sidebar.radio("Select Model", ["Random Forest", "Linear Regression"])

if model_choice == "Random Forest":
    model = RandomForestRegressor(n_estimators=500, max_depth=6, random_state=42)
else:
    model = LinearRegression()

# Train Model
model.fit(X_train, y_train)
prediction = model.predict(df)

# Display Prediction
st.subheader("Predicted Calories Burned")
st.metric(label="Calories Burned", value=f"{prediction[0]:.2f} kcal")

# Performance Metrics
st.subheader("Model Performance")
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
st.write(f"**MAE:** {mae:.2f}")
st.write(f"**RMSE:** {rmse:.2f}")

# Visualizations
st.subheader("Calories Burned Distribution")
fig, ax = plt.subplots(figsize=(8, 4))
sns.histplot(data["Calories"], bins=20, kde=True, ax=ax, color="skyblue")
plt.axvline(x=prediction[0], color='red', linestyle='--', label="Your Prediction")
plt.legend()
st.pyplot(fig)

# Show Similar Data Entries
st.subheader("Similar Entries in Dataset")
similar_data = data[(data["Calories"] >= prediction[0] - 10) & (data["Calories"] <= prediction[0] + 10)]
st.dataframe(similar_data.sample(5) if not similar_data.empty else "No similar entries found.", use_container_width=True)

# Calculate Percentage Comparisons
st.subheader("Comparison with Other Users")
st.write("How you compare with others in the dataset:")
st.write(f"You are older than {100 * (data['Age'] < df['Age'].values[0]).mean():.2f}% of other people.")
st.write(f"Your exercise duration is higher than {100 * (data['Duration'] < df['Duration'].values[0]).mean():.2f}% of other people.")
st.write(f"You have a higher heart rate than {100 * (data['Heart_Rate'] < df['Heart_Rate'].values[0]).mean():.2f}% of other people during exercise.")
st.write(f"You have a higher body temperature than {100 * (data['Body_Temp'] < df['Body_Temp'].values[0]).mean():.2f}% of other people during exercise.")

