import xgboost as xgb
import streamlit as st
import pandas as pd
import numpy as np
import time


#Loading up the Regression model we created
model = xgb.XGBRegressor()
model.load_model('xgb_model.json')

st.title("Car Price Prediction")

st.markdown("""
Created on July 1 2022
@author: M. Irvan Dimetrio
""")
st.markdown("""
This app will try to determine the exact price of the car based on user input on several variables or features. 
Using the ML model, namely **XGBoosting Regression. For the results of the ML model get a score of 92% CV(Cross-Validation).**

* You can check my jupyter notebook back end for data analysis in my github in here [Github](https://github.com/IrvanDimetrio/Kaggle-Project/blob/main/Car%20Price%20Prediction/Car%20Predic%20V3.ipynb).

* The dataset used to perform this prediction uses the dataset at [Kaggle](https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho). Consists of 6717 rows and 13 columns.
""")

html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Car Price Prediction App </h2>
    </div><br><br><br>
    """
st.markdown(html_temp,unsafe_allow_html=True)

# Define the prediction function
@st.cache
def predict(km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats,vehicle_age):

    if owner == 'First Owner':
        owner = 0
    elif owner == 'Second Owner':
        owner = 1
    elif owner == 'Third Owner':
        owner = 2
    elif owner == 'Fourth & Above Owner':
        owner = 3
    
    if fuel == 'Diesel':
        fuel = 0
    elif fuel == 'Petrol':
        fuel= 1
    
    if seller_type == 'Individual':
        seller_type = 0
    elif seller_type == 'Dealer':
        seller_type = 1
    elif seller_type == 'Trustmark Dealer':
        seller_type = 2
    
    if transmission == 'Automatic':
        transmission = 0
    elif transmission == 'Manual':
        transmission = 1
    
    
    prediction = model.predict(pd.DataFrame([[km_driven,fuel,seller_type,transmission,owner,mileage,engine,max_power,seats,vehicle_age]], 
    columns=['km_driven','fuel','seller_type','transmission','owner','mileage','engine','max_power','seats','vehicle_age']))

    print(prediction)
    return prediction

def main():

    # year = st.number_input('When was the year of your vehicle purchase? (Example : 2017)', min_value = 1994, max_value = 2022)
    vehicle_age = st.slider('When was your last vehicle purchased? (Example : 2017)', 1994, 2020)
    vehicle_age = 2022 - vehicle_age
    vehicle_age2 = np.log(vehicle_age)

    km_driven = st.number_input('How many kilometers does your vehicle have? (Example : 1000 Km)', min_value = 1)
    km_driven2 = np.log(km_driven)

    fuel = st.selectbox(
     'What is the fuel type of your vehicle?',
     ('Diesel', 'Petrol'))
    seller_type =  st.selectbox(
     'What type of seller are you?',
     ('Individual', 'Dealer', 'Trustmark Dealer'))
    transmission = st.selectbox(
     'What is the transmission type of your vehicle?',
     ('Automatic', 'Manual'))
    owner = st.selectbox(
     'How many previous owners have there been?', 
     ('First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner'))

    mileage = st.number_input('How many kilometers per gallon (kmpl) does your vehicle get? (Example : 17,7 kmpl)', min_value = 1.0, max_value = 42.0)

    engine = st.number_input('What is the capacity of your vehicle? (Example : 1298 CC)', min_value = 1, max_value = 3604)

    max_power = st.number_input('What is the max power (bhp) of your vehicle? (Example : 90 bhp)', min_value = 1.0)
    max_power2 = np.log(max_power)

    seats = st.number_input(
     'How many seats does your vehicle have? (Example : 5 seats)' , min_value = 2, max_value = 14)

    if st.button("Predict"):
        time.sleep(2)
        result=predict(km_driven2,fuel,seller_type,transmission,owner,mileage,engine,max_power2,seats,vehicle_age2)
        end_result = np.exp(result)
        # st.success('The predicted price of the car is {} Rupee India'.format(end_result))
        st.success('The predicted price of the car is ')
        st.subheader(f'₹{end_result[0]:.2f} Rupee India') 

if __name__=='__main__':
    main()
       
