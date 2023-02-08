#streamlit application that displays the results of the model
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="streamlit_main",
    page_icon="👋",
)


st.title('Kullanılan 4 modelin özet bilgileri')

st.write('TfIdf Train Shape (55858, 1, 89697)')
st.write('TfIdf Validation Shape (6983, 1, 89697) ')
st.write('TfIdf Test Shape (6982, 1, 89697)')


st.write("------------------------")

st.write('Wavelet Train Shape (55858, 1, 44849)')
st.write('Wavelet Validation Shape(6983, 1, 44849) ')
st.write('Wavelet Test Shape (6982, 1, 44849)')


st.write("------------------------")

st.write('Y Train Shape (55858, 3)')
st.write('Y Validation Shape (6983, 3) ')
st.write('Y Test Shape (6982, 3)')

st.subheader("train parameters")
st.code('''
early_stopping = EarlyStopping(monitor='val_loss', verbose=1 ,patience=1, mode='min')


#train model
history = model_tf2.fit(X_train, y_train,
                    batch_size=128,
                    epochs=5,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    callbacks=[early_stopping])
 ''')

st.subheader('model 1 ')
st.code(''' model_tf = Sequential()
model_tf.add(Conv1D(128, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf.add(MaxPooling1D(1))
model_tf.add(Dropout(0.5))
model_tf.add(Conv1D(64, 1, activation='relu'))
model_tf.add(MaxPooling1D(1))
model_tf.add(Dropout(0.5))
model_tf.add(Conv1D(32, 1, activation='relu'))
model_tf.add(MaxPooling1D(1))
model_tf.add(Flatten())
model_tf.add(Dense(3, activation='softmax'))

model_tf.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf.summary()

   
''')
model= pd.DataFrame({'tfidf loss' :0.1835 , 'tfidf accuracy':  0.9465,"wavelet loss":0.1982 , "wavelet accuracy": 0.9407},index=[0])

st.dataframe(model)

st.subheader('model 2 ')
st.code(''' model_tf2 = Sequential()
model_tf2.add(Conv1D(64, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf2.add(MaxPooling1D(1))
model_tf2.add(Dropout(0.5))
model_tf2.add(Conv1D(128, 1, activation='relu'))
model_tf2.add(MaxPooling1D(1))
model_tf2.add(Flatten())
model_tf2.add(Dense(3, activation='softmax'))

model_tf2.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf2.summary()

   
''')
model2= pd.DataFrame({'tfidf loss' :0.172 , 'tfidf accuracy': 0.9491,"wavelet loss":0.1893 , "wavelet accuracy": 0.9412},index=[0])

st.dataframe(model2)

st.subheader('model 3')
st.code(''' model_tf3 = Sequential()
model_tf3.add(Conv1D(128, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf3.add(MaxPooling1D(1))
model_tf3.add(Dropout(0.5))
model_tf3.add(Conv1D(32, 1, activation='relu'))
model_tf3.add(MaxPooling1D(1))
model_tf3.add(Flatten())
model_tf3.add(Dense(3, activation='softmax'))

model_tf3.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf3.summary()
     
   
''')
model3= pd.DataFrame({'tfidf loss' : 0.1700 , 'tfidf accuracy': 0.9480,"wavelet loss":0.1859 , "wavelet accuracy": 0.9411},index=[0])
  
st.dataframe(model3)


st.subheader('model 4')
st.code(''' model_tf4 = Sequential()
model_tf4.add(Conv1D(64, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf4.add(MaxPooling1D(1))
model_tf4.add(Flatten())
model_tf4.add(Dropout(0.5))
model_tf4.add(Dense(128, activation='relu'))
model_tf4.add(Dense(3, activation='softmax'))

model_tf4.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf4.summary()
     
   
''')
model4= pd.DataFrame({'tfidf loss' : 0.1751 , 'tfidf accuracy': 0.9455,"wavelet loss":0.1870 , "wavelet accuracy": 0.9415},index=[0])
     #show dataframe

st.dataframe(model4)
