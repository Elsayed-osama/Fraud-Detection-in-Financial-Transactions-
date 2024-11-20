import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder, StandardScaler
with open("Fraud_Detection.sav", "rb") as model_file:
    model = pickle.load(model_file)
data = pd.read_csv("Online Payment Fraud Detection.csv")
st.title("Online Payment Fraud Detection")
st.write("Enter the details to check for potential fraud")
step_input = st.number_input('Step', min_value=0, step=1)
type_input = st.selectbox('Type', data['type'].unique())
amount_input = st.number_input('Amount', min_value=0.0)
oldbalanceOrg_input = st.number_input('Old Balance Org', min_value=0.0)
oldbalanceDest_input = st.number_input('Old Balance Dest', min_value=0.0)
type_mapping = {'PAYMENT': 3, 'TRANSFER': 4, 'CASH_OUT': 1, 'DEBIT': 2, 'CASH_IN': 0}
type_encoded = type_mapping[type_input]
input_data = pd.DataFrame({
    'step': [step_input],
    'type': [type_encoded],
    'amount': [amount_input],
    'oldbalanceOrg': [oldbalanceOrg_input],
    'oldbalanceDest': [oldbalanceDest_input]
})
with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
input_data_scaled = scaler.transform(input_data)
if st.button('Predict'):
    prediction = model.predict(input_data_scaled)
    if prediction == 1:
        st.error("Warning: This transaction is potentially fraudulent!")
        st.image("https://media.licdn.com/dms/image/C5612AQHDBQHTY2oiHQ/article-cover_image-shrink_600_2000/0/1545048419916?e=2147483647&v=beta&t=OADhMFUObqr3o2GsLUqI5RFV8upozAUyawPrsZU47Tk")
    else:
        st.success("This transaction seems legitimate.")
        st.image("https://media.licdn.com/dms/image/C4E12AQF2tCnouOrupg/article-cover_image-shrink_600_2000/0/1563183516700?e=2147483647&v=beta&t=laASHx98-yIAX-5YDgxFi6j-yGRwCoJjxjmYeQ_8xx0")
