import numpy as np 
import keras 
import streamlit as st 
import pandas as pd 
import pickle
from pathlib import Path

def read_markdown_file(markdown_file):
    return Path(markdown_file).read_text()

intro_markdown = read_markdown_file("main.md")
st.markdown(intro_markdown, unsafe_allow_html=True)
scaler = pickle.load(open('min_max_scaler.pkl', 'rb'))
# df = pd.DataFrame(columns =['Pollution'])

input_data_list = []

day_1 = st.slider('Hr1 Pollution Level', 0, 400, 25)
input_data_list.append([day_1])
day_2 = st.slider('Hr2 Pollution Level', 0, 400, 25)
input_data_list.append([day_2])
day_3 = st.slider('Hr3 Pollution Level', 0, 400, 25)
input_data_list.append([day_3])
day_4 = st.slider('Hr4 4 Pollution Level', 0, 400, 25)
input_data_list.append([day_4])
day_5 = st.slider('Hr5 Pollution Level', 0, 400, 25)
input_data_list.append([day_5])


    
next_day_predicion = st.button("Predict for Next Hour.")

if next_day_predicion:
    reconstructed_model = keras.models.load_model("model/air_pollution_forecasting_model")
    model = reconstructed_model
    df = pd.DataFrame(input_data_list, columns= ['Pollution'])
    print(df)
    to_be_predicted = scaler.transform(df.values)
    to_be_predicted = to_be_predicted.reshape(1, 5, 1)
    Y_pred = np.round(model.predict(to_be_predicted),2)
    Y_pred = scaler.inverse_transform(Y_pred)
    st.text("The Predicted pollution for next day based on previous 5 day data is: ")
    st.text(str(Y_pred))
    print(int(Y_pred[0][0]))
    print(input_data_list)
    input_data_list[4] = [int(Y_pred[0][0])]
    print(input_data_list)
