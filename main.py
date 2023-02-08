#streamlit application that displays the results of the model
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt



st.set_page_config(
    page_title="streamlit_main",
    page_icon="👋",
)





st.title('Sentiment analizlere göre yapılan Time Series çalışmasının en iyi metriklere sahip modellerinin karşılaştırılması')


allscores = pd.read_csv("data/all_results.csv",index_col=0)
st.subheader('Time Series Analysis')

st.subheader('Kullanılan 3 farklı modelin özet bilgileri')

st.subheader('model 1 : LSTM')
st.code(''' rbct_lstm = Sequential()
rbct_lstm.add(LSTM(64, return_sequences=True, input_shape = (trainX.shape[1],trainX.shape[2])))

rbct_lstm.add(LSTM(64,return_sequences=False))
rbct_lstm.add(Dense(25))
rbct_lstm.add(Dense(1))


rbct_lstm.summary()
   
''')

st.subheader('model 2 : Conv1D')
st.code(''' 
rbct_conv1d = Sequential()
rbct_conv1d.add(InputLayer((trainX.shape[1],trainX.shape[2])))
rbct_conv1d.add(Conv1D(64, kernel_size=2))
rbct_conv1d.add(Flatten())
rbct_conv1d.add(Dense(8, 'linear'))
rbct_conv1d.add(Dense(1, 'linear'))

rbct_conv1d.summary()
   
''')

st.subheader('model 3: Conv1D + LSTM')
st.code(''' rbct_convlstm = Sequential()
rbct_convlstm.add(InputLayer((trainX.shape[1],trainX.shape[2])))
rbct_convlstm.add(Conv1D(64, kernel_size=2))
rbct_convlstm.add(LSTM(64,return_sequences=False))
rbct_convlstm.add(Dense(8, 'linear'))
rbct_convlstm.add(Dense(1, 'linear'))

rbct_convlstm.summary()
     
   
''')


st.subheader('En iyi çalışan 2 model')

st.write(allscores[allscores["r2"]==allscores["r2"].max()])


st.write(allscores[(allscores["mae"]==allscores["mae"].min())])




st.subheader('En iyi 2 modelin tahmin grafikleri')
st.image('img/fbct_hyb_ts_pred.png')
st.image('img/rbct_cnn_ts_pred.png')
