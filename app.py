import numpy as np
import streamlit as st
import joblib
import pandas as pd

model = joblib.load('flight_model.pkl')
encoders = joblib.load("encoders.pkl")
df = pd.read_csv("./flight_data.csv")

def app():
    st.title('Flight Delay Prediction')
    
    carrier = st.selectbox('Carrier', sorted(df['carrier'].unique()))
    origin = st.selectbox('Origin', sorted(df['origin'].unique()))
    dest = st.selectbox('Destination', sorted(df['dest'].unique()))
    distance = st.slider('Distance (miles)', 0,20000)
    hour = st.slider('Scheduled Departure Hour (0-23)', 0,23)
    day = st.slider('Scheduled Departure Day (1-31)', 0,31)
    month = st.slider('Scheduled Departure Month (1-12)',0,12)

    if st.button('Predict'):
        input_data = pd.DataFrame({'carrier': [carrier],'origin': [origin],'dest': [dest],'distance': [distance],'hour': [hour],'day': [day],'month': [month] })
        for col in encoders.keys():
            input_data[col] = encoders[col].transform(input_data[col])[0]

        prediction = model.predict(input_data.values)
        if prediction[0] == 1:
            st.write('This flight is likely to be departing late.Thank You for your Cooperation.')
        else:
            st.write('This flight is likely to be departing on time.')


if __name__ == '__main__':
    app()