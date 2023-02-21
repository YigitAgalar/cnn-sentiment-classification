#streamlit application that displays the results of the model
import streamlit as st
import pandas as pd



st.set_page_config(
    page_title="streamlit_main",
    page_icon="👋",
)


st.title('Optimize edilen modellerin karşılaştırılması ( chi2 vs wavelet)') 

st.write('Chi2 Train Shape (55858, 1, 44849)')
st.write('Chi2 Validation Shape(6983, 1, 44849) ')
st.write('Chi2 Test Shape (6982, 1, 44849)')


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
early_stopping = EarlyStopping(monitor='val_loss', verbose=1 ,patience=2, mode='min',restore_best_weights=True)

#train model
history = model_tf1.fit(X_train, y_train,
                    batch_size=128,
                    epochs=10,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    callbacks=[early_stopping])
 ''')

st.subheader('model 1')
st.code('''
odel: "sequential_12"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_17 (LSTM)              (None, 1, 1024)           187899904 
                                                                 
 dropout_9 (Dropout)         (None, 1, 1024)           0         
                                                                 
 lstm_18 (LSTM)              (None, 128)               590336    
                                                                 
 dense_8 (Dense)             (None, 3)                 387       
                                                                 
=================================================================s
Total params: 188,490,627
Trainable params: 188,490,627
Non-trainable params: 0
_________________________________________________________________

''')


st.code('''{
'units': 1024,
 'activation': 'relu',
 'n_layers': 0,
 'dropout_last': 0.9,
 'lstm_units_last': 128,
 'learning_rate': 0.001,
 }''')
model1= pd.DataFrame({'chi2 loss' :0.1144 , 'chi2 acc': 0.9613},index=[0])
model1wv = pd.DataFrame({"wv loss":0.1800 , "wv acc": 0.9434},index=[0])
st.write("model1 chi2 input graphs")
st.image('chi_img/chi1_full.png')
st.write("model1 wavelet input graphs")
st.image('chi_img/chi_ww1_full.png')
st.dataframe(model1)
st.dataframe(model1wv)


st.subheader('model 2')
st.code(''' Model: "sequential_11"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_15 (LSTM)              (None, 1, 128)            23028736  
                                                                 
 dropout_8 (Dropout)         (None, 1, 128)            0         
                                                                 
 lstm_16 (LSTM)              (None, 8)                 4384      
                                                                 
 dense_7 (Dense)             (None, 3)                 27        
                                                                 
=================================================================
Total params: 23,033,147
Trainable params: 23,033,147
Non-trainable params: 0
_________________________________________________________________
   
''')

st.code('''{
'units': 128,
 'activation': 'relu',
 'n_layers': 0,
 'dropout_last': 0.5,
 'lstm_units_last': 8,
 'learning_rate': 0.001
 }''')


model2= pd.DataFrame({'chi2 loss' :0.1473 , 'chi2 acc': 0.9557},index=[0])
model2wv = pd.DataFrame({"wv loss":0.1859 , "wv acc": 0.9427},index=[0])
st.write("model2 chi2 input graphs")
st.image('chi_img/chi2_full.png')


st.write("model2 wavelet input graphs")
st.image('chi_img/chi_ww2_full.png')
st.dataframe(model2)
st.dataframe(model2wv)


st.subheader('model 3')
st.code(''' Model: "sequential_13"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_19 (LSTM)              (None, 1, 128)            23028736  
                                                                 
 dropout_10 (Dropout)        (None, 1, 128)            0         
                                                                 
 lstm_20 (LSTM)              (None, 1, 256)            394240    
                                                                 
 dropout_11 (Dropout)        (None, 1, 256)            0         
                                                                 
 lstm_21 (LSTM)              (None, 64)                82176     
                                                                 
 dense_9 (Dense)             (None, 3)                 195       
                                                                 
=================================================================
Total params: 23,505,347
Trainable params: 23,505,347
Non-trainable params: 0
_________________________________________________________________
''')

st.code('''{
 'units': 128,
 'activation': 'sigmoid',
 'n_layers': 1,
 'dropout_last': 0.7000000000000001,
 'lstm_units_last': 64,
 'learning_rate': 0.01,
 'dropout0': 0.4,
 'lstm_units0': 256,
 }''')

model3= pd.DataFrame({'chi2 loss' :0.1612 , 'chi2 acc': 0.9525},index=[0])
model3wv = pd.DataFrame({"wv loss":0.1824 , "wv acc": 0.9395},index=[0])
st.write("model3 chi2 input graphs")
st.image('chi_img/chi3_full.png')


st.write("model3 wavelet input graphs")
st.image('chi_img/chi_ww3_full.png')
st.dataframe(model3)
st.dataframe(model3wv)



st.subheader('model 4')
st.code(''' Model: "sequential_14"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_22 (LSTM)              (None, 1, 32)             5744896   
                                                                 
 dropout_12 (Dropout)        (None, 1, 32)             0         
                                                                 
 lstm_23 (LSTM)              (None, 1, 512)            1116160   
                                                                 
 dropout_13 (Dropout)        (None, 1, 512)            0         
                                                                 
 lstm_24 (LSTM)              (None, 16)                33856     
                                                                 
 dense_10 (Dense)            (None, 3)                 51        
                                                                 
=================================================================
Total params: 6,894,963
Trainable params: 6,894,963
Non-trainable params: 0
_________________________________________________________________
''')

st.code('''{
 'units': 32,
 'activation': 'tanh',
 'n_layers': 1,
 'dropout_last': 0.7000000000000001,
 'lstm_units_last': 16,
 'learning_rate': 0.001,
 'dropout0': 0.5,
 'lstm_units0': 512,
 }''')
 
model4= pd.DataFrame({'chi2 loss' :0.1616 , 'chi2 acc': 0.9508},index=[0])
model4wv = pd.DataFrame({"wv loss":0.1952 , "wv acc": 0.9338},index=[0])
st.write("model4 chi2 input graphs")
st.image('chi_img/chi4_full.png')

st.write("model4 wavelet input graphs")
st.image('chi_img/chi_ww4_full.png')
st.dataframe(model4)
st.dataframe(model4wv)


st.subheader('model 5')
st.code(''' Model: "sequential_15"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 lstm_25 (LSTM)              (None, 1, 1024)           187899904 
                                                                 
 dropout_14 (Dropout)        (None, 1, 1024)           0         
                                                                 
 lstm_26 (LSTM)              (None, 1, 256)            1311744   
                                                                 
 dropout_15 (Dropout)        (None, 1, 256)            0         
                                                                 
 lstm_27 (LSTM)              (None, 1, 64)             82176     
                                                                 
 dropout_16 (Dropout)        (None, 1, 64)             0         
                                                                 
 lstm_28 (LSTM)              (None, 256)               328704    
                                                                 
 dense_11 (Dense)            (None, 3)                 771       
                                                                 
=================================================================
Total params: 189,623,299
Trainable params: 189,623,299
Non-trainable params: 0
_________________________________________________________________
''')


st.code('''{
'units': 1024,
 'activation': 'relu',
 'n_layers': 2,
 'dropout_last': 0.30000000000000004,
 'lstm_units_last': 256,
 'learning_rate': 0.001,
 'dropout0': 0.1,
 'lstm_units0': 256,
 'dropout1': 0.4,
 'lstm_units1': 64,
 'dropout2': 0.1,
 }''')

model5= pd.DataFrame({'chi2 loss' :0.1650 , 'chi2 acc': 0.9490},index=[0])
model5wv = pd.DataFrame({"wv loss":0.1966 , "wv acc": 0.9369},index=[0])
st.write("model5 chi2 input graphs")
st.image('chi_img/chi5_full.png')

st.write("model5 wavelet input graphs")
st.image('chi_img/chi_ww5_full.png')
st.dataframe(model5)
st.dataframe(model5wv)


