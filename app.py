import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import base64

# Load and preprocess the dataset
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)
    return data

# Function to add background image with proper fitting and resolution
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url(data:image/jpeg;base64,{encoded}) no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, .stApp p {{
            color: #000000 !important;
            font-weight: bold;
        }}
        .stApp .stButton>button {{
            background-color: #4CAF50; /* Green */
            border: none;
            color: white;
            padding: 10px 24px;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 16px;
            margin: 4px 2px;
            cursor: pointer;
        }}
        .stApp .stTextInput>div>input {{
            border: 2px solid #000;
            padding: 10px;
        }}
        .stApp .stTextInput>div>label {{
            font-weight: bold;
            color: #000;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Load data and preprocess
data = load_data()
X = data.drop('Outcome', axis=1)
y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Add background image
add_bg_from_local('assets/background1.jpg')

# Streamlit UI
st.title("Obesity Prediction App")
st.write("Enter the details below to predict if you have obesity or not.")

# User input
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0, step=1)
glucose = st.number_input('Glucose Level', min_value=0, max_value=200, value=120, step=1)
blood_pressure = st.number_input('Blood Pressure (mm Hg)', min_value=0, max_value=150, value=80, step=1)
skin_thickness = st.number_input('Skin Thickness (mm)', min_value=0, max_value=100, value=20, step=1)
insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80, step=1)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0, step=0.1)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=3.0, value=0.5, step=0.01)
age = st.number_input('Age', min_value=0, max_value=120, value=25, step=1)

# Prediction
user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
user_data_scaled = scaler.transform(user_data)
prediction = model.predict(user_data_scaled)
result = "You have obesity." if prediction[0] == 1 else "You do not have obesity."

# Display result
if st.button('Predict'):
    st.subheader("Prediction Result:")
    st.markdown(f"<h1 style='color:red;'>{result}</h1>", unsafe_allow_html=True)
