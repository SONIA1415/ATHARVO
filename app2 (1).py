import streamlit as st
import pandas as pd
from joblib import load
import matplotlib.pyplot as plt
model = load('random_forest_model.joblib')
label_encoder = load('label_encoder.joblib')
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;  /* Light grey background */
        color: #000000;  /* Black text color for better readability */
    }
    .sidebar .sidebar-content {
        background-color: #e0e0e0;  /* Slightly lighter grey for sidebar */
        color: #000000;  /* Black text color for sidebar */
    }
    h1, h2, h3, p {
        color: #000000;  /* Black text color */
        font-size: 20px;  /* Font size 20px */
        font-family: 'Arial', sans-serif;  /* Clean font family */
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.title('Purchase Prediction App')


st.sidebar.header('User Input')
st.sidebar.write("### Enter User Details")

def user_input_features():
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
    age = st.sidebar.slider('Age', 18, 100, 25)
    estimated_salary = st.sidebar.slider('Estimated Salary', 10000, 150000, 50000)
    return pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'EstimatedSalary': [estimated_salary]
    })

df = user_input_features()


df['Gender'] = label_encoder.transform(df['Gender'])


prediction = model.predict(df)
prediction_proba = model.predict_proba(df)


st.write("## Prediction Results")

if prediction[0] == 1:
    st.markdown('<h2 style="color: #4caf50;">The user is likely to make a purchase.</h2>', unsafe_allow_html=True)  # Green
else:
    st.markdown('<h2 style="color: #f44336;">The user is not likely to make a purchase.</h2>', unsafe_allow_html=True)  # Red
st.write("### Prediction Probability")
fig, ax = plt.subplots(figsize=(8, 4))  
categories = ['Purchase', 'No Purchase']
probs = prediction_proba[0]

ax.bar(categories, probs, color='#ff5722')  
ax.set_ylim(0, 1)
ax.set_ylabel('Probability', fontsize=18, color='#000000')
ax.set_title('Probability of Purchase', fontsize=18, color='#000000')
ax.set_facecolor('#f5f5f5')  
ax.tick_params(axis='both', colors='#000000')  

st.pyplot(fig)
st.write("### User Input Data")
st.write(df)
