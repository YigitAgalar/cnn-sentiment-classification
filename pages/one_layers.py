import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




st.subheader('1024')
st.code('''
model_1l_chi1 = Sequential()
model_1l_chi1.add(LSTM(1024,return_sequences=False, activation="relu", input_shape=(1,Wv_train.shape[2])))
            
model_1l_chi1.add(Dropout(0.9))
model_1l_chi1.add(Dense(3, activation='softmax'))

model_1l_chi1.compile(loss='categorical_crossentropy',
            optimizer=Adam(0.001),
            metrics=['accuracy'])
    

''')

st.subheader('Graphs')
st.write("Chi")
st.image('onelayer_imgs/chi_1024_full.png')
st.write("Wavelet")
st.image('onelayer_imgs/wv_1024_full.png')

model3wv = pd.DataFrame({"wv loss":0.1750 , "wv acc": 0.9445},index=[0])
model3chi = pd.DataFrame({"chi loss":0.1496 , "chi acc": 0.9557},index=[0])
st.dataframe(model3wv)
st.dataframe(model3chi)


st.subheader('512')
st.code('''
model_1l_chi2 = Sequential()
model_1l_chi2.add(LSTM(512,return_sequences=False, activation="relu", input_shape=(1,Wv_train.shape[2])))
            
model_1l_chi2.add(Dropout(0.9))
model_1l_chi2.add(Dense(3, activation='softmax'))

model_1l_chi2.compile(loss='categorical_crossentropy',
            optimizer=Adam(0.001),
            metrics=['accuracy'])
    
''')


st.subheader('Graphs')
st.write("Chi")
st.image('onelayer_imgs/chi_512_full.png')
st.write("Wavelet")
st.image('onelayer_imgs/wv_512_full.png')

model3wv = pd.DataFrame({"wv loss":0.1712 , "wv acc": 0.9434},index=[0])
model3chi = pd.DataFrame({"chi loss":0.1486 , "chi acc": 0.9554},index=[0])
st.dataframe(model3wv)
st.dataframe(model3chi)


st.subheader('256')
st.code('''
model_1l_chi3 = Sequential()
model_1l_chi3.add(LSTM(256,return_sequences=False, activation="relu", input_shape=(1,Wv_train.shape[2])))
            
model_1l_chi3.add(Dropout(0.9))
model_1l_chi3.add(Dense(3, activation='softmax'))

model_1l_chi3.compile(loss='categorical_crossentropy',
            optimizer=Adam(0.001),
            metrics=['accuracy'])

model_1l_chi3.summary()
    
''')



st.subheader('Graphs')
st.write("Chi")
st.image('onelayer_imgs/chi_256_full.png')
st.write("Wavelet")
st.image('onelayer_imgs/wv_256_full.png')

model3wv = pd.DataFrame({"wv loss":0.1760 , "wv acc": 0.9429},index=[0])
model3chi = pd.DataFrame({"chi loss":0.1538 , "chi acc": 0.9557},index=[0])
st.dataframe(model3wv)
st.dataframe(model3chi)
