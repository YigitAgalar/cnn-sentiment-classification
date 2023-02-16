import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.set_page_config(
    page_title="streamlit_main",
    page_icon="👋",
)

st.header("16-02-2023")

st.subheader('LSTM Model 1')
st.code(''' 
model_lstm_ww1=Sequential()
model_lstm_ww1.add(LSTM(192, return_sequences=True,activation="relu", input_shape=(1,Wv_train.shape[2])))
model_lstm_ww1.add(Dropout(0.2))
model_lstm_ww1.add(LSTM(192,return_sequences=False,activation="relu"))
model_lstm_ww1.add(Dense(3,activation="softmax"))

model_lstm_ww1.summary()

model_lstm_ww1.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

                    

''')


st.subheader('LSTM1 Wavelet graphs')
st.image('lstm_img/lstm_ww1_full.png')
model1 = pd.DataFrame({"wv loss":0.1418 , "wv acc": 0.9402},index=[0])
st.dataframe(model1)

st.subheader('LSTM Model 2')
st.code(''' 
model_lstm_ww2 = Sequential()
model_lstm_ww2.add(LSTM(160, return_sequences=True,activation="relu", input_shape=(1,Wv_train.shape[2])))
model_lstm_ww2.add(Dropout(0.2))
model_lstm_ww2.add(LSTM(96,return_sequences=False,activation="relu"))
model_lstm_ww2.add(Dense(3,activation="softmax"))

model_lstm_ww2.summary()

model_lstm_ww2.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy']))

                    

''')


st.subheader('LSTM2 Wavelet graphs')
st.image('lstm_img/lstm_ww2_full.png')
model2 = pd.DataFrame({"wv loss":0.1795 , "wv acc": 0.9412},index=[0])


st.subheader('LSTM Model 3')
st.code(''' 
model_lstm_ww3 = Sequential()
model_lstm_ww3.add(LSTM(224, return_sequences=True,activation="relu", input_shape=(1,Wv_train.shape[2])))
model_lstm_ww3.add(Dropout(0.2))
model_lstm_ww3.add(LSTM(32,return_sequences=False,activation="relu"))
model_lstm_ww3.add(Dense(3,activation="softmax"))

model_lstm_ww3.summary()

model_lstm_ww3.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])

''')


st.subheader('LSTM3 Wavelet graphs')
st.image('lstm_img/lstm_ww3_full.png')
model3 = pd.DataFrame({"wv loss":0.1846 , "wv acc": 0.9392},index=[0])
st.dataframe(model3)



st.subheader('LSTM Model 4')
st.code(''' 
model_lstm_ww4 = Sequential()
model_lstm_ww4.add(LSTM(192, return_sequences=True,activation="relu", input_shape=(1,Wv_train.shape[2])))
model_lstm_ww4.add(Dropout(0.2))
model_lstm_ww4.add(LSTM(96,return_sequences=False,activation="relu"))
model_lstm_ww4.add(Dense(3,activation="softmax"))

model_lstm_ww4.summary()

model_lstm_ww4.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


''')


st.subheader('LSTM4 Wavelet graphs')
st.image('lstm_img/lstm_ww4_full.png')
model4 = pd.DataFrame({"wv loss":0.1790 , "wv acc": 0.9415},index=[0])
st.dataframe(model4)



st.subheader('LSTM Model 5')
st.code(''' 
model_lstm_ww5 = Sequential()
model_lstm_ww5.add(LSTM(224, return_sequences=True,activation="relu", input_shape=(1,Wv_train.shape[2])))
model_lstm_ww5.add(Dropout(0.2))
model_lstm_ww5.add(LSTM(96,return_sequences=True,activation="relu"))
model_lstm_ww5.add(Dropout(0.2))
model_lstm_ww5.add(LSTM(160,return_sequences=True,activation="relu"))
model_lstm_ww5.add(Dropout(0.2))
model_lstm_ww5.add(LSTM(224,return_sequences=False,activation="relu"))
model_lstm_ww5.add(Dense(3,activation="softmax"))

model_lstm_ww5.summary()

model_lstm_ww5.compile(loss='categorical_crossentropy',
                    optimizer='adam',
                    metrics=['accuracy'])


''')


st.subheader('LSTM5 Wavelet graphs')
st.image('lstm_img/lstm_ww5_full.png')
model5 = pd.DataFrame({"wv loss":0.1929 , "wv acc": 0.9382},index=[0])
st.dataframe(model5)