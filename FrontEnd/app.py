
import streamlit as st
import pandas as pd
import requests

st.title("Tip Prediction App")

st.title("Please enter the details to get tip amount")

# input data
total_bill = st.number_input("Total Bill", min_value = 0)
sex = st.selectbox("Sex", options = ["Male", "Female"])
smoker = st.selectbox("Smoker", options = ["Yes", "No"])
day = st.selectbox("Day", options = ["Thur", "Fri", "Sat", "Sun"])
time = st.selectbox("time", options = ["Lunch", "Dinner"]) 
size = st.number_input("size", min_value = 1, max_value = 10)

if st.button("Prediction"):
    input_data = {"total_bill": total_bill,
                  "sex": sex,
                  "day": day,
                  "time": time,                
                  "size": size,
                  "smoker": smoker}
    response = requests.post("http://127.0.0.1:5000/predict", json=input_data)

    if response.status_code == 200:
        prediction = response.json().get("prediction")
        st.write("The tip Value is", prediction)
    else: 
        st.write ("Error in prediction")    