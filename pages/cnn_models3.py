import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



st.header("16-02-2023")

st.subheader('64-32')
st.code('''
model_wv7d = Sequential()
model_wv7d.add(Conv1D(64, 3, activation='relu', input_shape=(Wv_train.shape[1],1)))
model_wv7d.add(MaxPooling1D(1))
model_wv7d.add(Dropout(0.5))
model_wv7d.add(Conv1D(32,3, activation='relu'))
model_wv7d.add(MaxPooling1D(1))
model_wv7d.add(Flatten())
model_wv7d.add(Dense(3, activation='softmax'))

model_wv7d.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_wv7d.summary()

''')

st.subheader('CNN1 Wavelet graphs')
st.image('cnn_img/model_cnn1_full.png')

model3wv = pd.DataFrame({"wv nbn loss":0.2250 , "wv nbn acc": 0.92580},index=[0])
st.dataframe(model3wv)


st.subheader('128-32')
st.code('''
model_wv_yeni2 = Sequential()
model_wv_yeni2.add(Conv1D(128, 1, activation='relu', input_shape=(Wv_train.shape[1],1)))
model_wv_yeni2.add(MaxPooling1D(1))
model_wv_yeni2.add(Dropout(0.5))
model_wv_yeni2.add(Conv1D(32, 1, activation='relu'))
model_wv_yeni2.add(MaxPooling1D(1))
model_wv_yeni2.add(Flatten())
model_wv_yeni2.add(Dense(3, activation='softmax'))

model_wv_yeni2.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_wv_yeni2.summary()

''')

st.subheader('CNN2 Wavelet graphs')
st.image('cnn_img/model_cnn2_full.png')

model3wv = pd.DataFrame({"wv nbn loss":0.2189 , "wv nbn acc": 0.9265},index=[0])
st.dataframe(model3wv)



st.subheader('128-64')
st.code('''modelwv = Sequential()
modelwv.add(Conv1D(128, 3, activation='relu', input_shape=(Wv_train.shape[1],1)))
modelwv.add(MaxPooling1D(3))
modelwv.add(Dropout(0.8))
modelwv.add(Conv1D(64,3, activation='relu'))
modelwv.add(Flatten())
modelwv.add(Dense(3, activation='softmax'))

modelwv.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

modelwv.summary()
''')


st.subheader('CNN3 Wavelet graphs')
st.image('cnn_img/model_cnn3_full.png')

model3wv = pd.DataFrame({"wv nbn loss":0.2303 , "wv nbn acc": 0.925236},index=[0])
st.dataframe(model3wv)