import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="streamlit_main",
    page_icon="👋",
)

st.header("13-02-2023")

st.subheader('128-64-32 no batch normalization')
st.code('''
#build model
modelwv2 = Sequential()
modelwv2.add(Conv1D(128, 1, activation='relu', input_shape=(1,Wv_train.shape[2])))
modelwv2.add(MaxPooling1D(1))
modelwv2.add(Dropout(0.5))
modelwv2.add(Conv1D(64, 1, activation='relu'))
modelwv2.add(MaxPooling1D(1))
modelwv2.add(Dropout(0.5))
modelwv2.add(Conv1D(32, 1, activation='relu'))
modelwv2.add(MaxPooling1D(1))
modelwv2.add(Flatten())
modelwv2.add(Dense(3, activation='softmax'))

modelwv2.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

modelwv2.summary()

''')

st.subheader('128-64-32 batch normalization')
st.code('''
model_tf_bn = Sequential()
model_tf_bn.add(Conv1D(128, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf_bn.add(BatchNormalization())
model_tf_bn.add(MaxPooling1D(1))
model_tf_bn.add(Dropout(0.5))
model_tf_bn.add(Conv1D(64, 1, activation='relu'))
model_tf_bn.add(BatchNormalization())
model_tf_bn.add(MaxPooling1D(1))
model_tf_bn.add(Dropout(0.5))
model_tf_bn.add(Conv1D(32, 1, activation='relu'))
model_tf_bn.add(BatchNormalization())
model_tf_bn.add(MaxPooling1D(1))
model_tf_bn.add(Flatten())
model_tf_bn.add(Dense(3, activation='softmax'))

model_tf_bn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf_bn.summary()

''')

model3= pd.DataFrame({'tfidf loss' : 0.1609 , 'tfidf acc': 0.9419,'tfidf bn loss' : 0.1962 , 'tfidf bn acc': 0.9350},index=[0])
model3wv = pd.DataFrame({"wv nbn loss":0.1829 , "wv nbn acc": 0.9402,"wv bn loss":0.2224 , "wv bn acc": 0.9220},index=[0])

st.dataframe(model3)
st.dataframe(model3wv)

st.subheader('tfidf bn')
st.image('img_graph/12864_bn_full.png')

st.subheader('tfidf nbn')
st.image('img_graph/12864_nobn_full.png')

st.subheader('wv nbn')
st.image('img_graph/12864_wv_nobn_full.png')

st.subheader('wv bn')
st.image('img_graph/m2_wv_bn.png')




st.subheader('128-128-32 nobn')
st.code('''
#build model
model_tf7_nobn = Sequential()
model_tf7_nobn.add(Conv1D(128, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf7.add(BatchNormalization())
model_tf7_nobn.add(MaxPooling1D(1))
model_tf7_nobn.add(Dropout(0.2))
model_tf7_nobn.add(Conv1D(128, 1, activation='relu'))
model_tf7.add(BatchNormalization())
model_tf7_nobn.add(MaxPooling1D(1))
model_tf7_nobn.add(Dropout(0.2))
model_tf7_nobn.add(Conv1D(32, 1, activation='relu'))
model_tf7.add(BatchNormalization())
model_tf7_nobn.add(MaxPooling1D(1))
model_tf7_nobn.add(Flatten())
model_tf7_nobn.add(Dense(3, activation='softmax'))

model_tf7_nobn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf7_nobn.summary()

''')



st.subheader('128-128-32 bn')
st.code('''
#build model
model_tf7_nobn = Sequential()
model_tf7_nobn.add(Conv1D(128, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf7_nobn.add(MaxPooling1D(1))
model_tf7_nobn.add(Dropout(0.2))
model_tf7_nobn.add(Conv1D(128, 1, activation='relu'))
model_tf7_nobn.add(MaxPooling1D(1))
model_tf7_nobn.add(Dropout(0.2))
model_tf7_nobn.add(Conv1D(32, 1, activation='relu'))
model_tf7_nobn.add(MaxPooling1D(1))
model_tf7_nobn.add(Flatten())
model_tf7_nobn.add(Dense(3, activation='softmax'))

model_tf7_nobn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf7_nobn.summary()

''')


