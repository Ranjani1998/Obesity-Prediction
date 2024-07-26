import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
import base64
import os

# Function to load dataset
@st.cache
def load_data():
    url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(url, names=columns)
    return data

# Function to add background image from local file
def add_bg_from_local(image_file):
    if os.path.isfile(image_file):
        with open(image_file, "rb") as image:
            encoded = base64.b64encode(image.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url(data:image/jpeg;base64,{encoded});
                background-size: cover;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error(f"File not found: {image_file}")

# Load the data
data = load_data()

# Sidebar
st.sidebar.title("Pima Indians Diabetes Dataset")
st.sidebar.write("This dataset is used to predict the onset of diabetes based on diagnostic measurements.")

# Add background image
add_bg_from_local('assets/background1.jpg')

# Display the data
if st.sidebar.checkbox("Show raw data"):
    st.subheader("Pima Indians Diabetes Dataset")
    st.write(data)

# Data Preprocessing
data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = data[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
data.fillna(data.median(), inplace=True)

# Feature Distribution
if st.sidebar.checkbox("Show feature distributions"):
    st.subheader("Feature Distributions")
    data.hist(bins=15, figsize=(15, 10), layout=(3, 3))
    st.pyplot(plt.gcf())

# Correlation Heatmap
if st.sidebar.checkbox("Show correlation heatmap"):
    st.subheader("Correlation Heatmap")
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    st.pyplot(plt.gcf())

# Pair Plot
if st.sidebar.checkbox("Show pair plot"):
    st.subheader("Pair Plot of Features")
    sns.pairplot(data, hue='Outcome', diag_kind='kde', markers=['o', 's'])
    st.pyplot(plt.gcf())

# Data Normalization
X = data.drop('Outcome', axis=1)
y = data['Outcome']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model Training
model = LogisticRegression()
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display model performance
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy}")
st.write("Classification Report:")
st.text(report)

# ROC Curve
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
st.pyplot(plt.gcf())

st.sidebar.write("Developed by Subaranjani T")
