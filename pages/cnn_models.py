#streamlit application that displays the results of the model
import streamlit as st
import pandas as pd
import numpy as np
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
early_stopping = EarlyStopping(monitor='val_loss', verbose=1 ,patience=2, mode='min',restore_best_weights=True)

#train model
history = model_tf1.fit(X_train, y_train,
                    batch_size=128,
                    epochs=5,
                    validation_data=(X_val, y_val),
                    verbose=1,
                    callbacks=[early_stopping])
 ''')

st.subheader("bn -> batch normalization  / nbn -> no batch normalization")

st.subheader('model 1 bn')
st.code('''
model_tf1_bn = Sequential()
model_tf1_bn.add(Conv1D(64, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf1_bn.add(BatchNormalization())
model_tf1_bn.add(MaxPooling1D(1))
model_tf1_bn.add(Dropout(0.5))
model_tf1_bn.add(Conv1D(128, 1, activation='relu'))
model_tf1_bn.add(BatchNormalization())
model_tf1_bn.add(MaxPooling1D(1))
model_tf1_bn.add(Flatten())
model_tf1_bn.add(Dense(3, activation='softmax'))

model_tf1_bn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf1_bn.summary()

''')
st.subheader('model 1 nbn')
st.code(''' model_tf1 = Sequential()
model_tf1.add(Conv1D(64, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf1.add(MaxPooling1D(1))
model_tf1.add(Dropout(0.5))
model_tf1.add(Conv1D(128, 1, activation='relu'))
model_tf1.add(MaxPooling1D(1))
model_tf1.add(Flatten())
model_tf1.add(Dense(3, activation='softmax'))

model_tf1.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf1.summary()

   
''')
model2= pd.DataFrame({'tfidf loss' :0.1652 , 'tfidf acc': 0.9457,'tfidf bn loss' :0.1902 , 'tfidf bn acc': 0.9338},index=[0])
model2wv = pd.DataFrame({"wv nbn loss":0.1770 , "wv nbn acc": 0.9437,"wavelet bn loss":0.2056 , "wv bn acc": 0.9281},index=[0])
st.subheader('model 1 nbn tfidf')
st.image('img_graph/m2_tf_nbn.png')
st.subheader('model 1 bn tfidf')
st.image('img_graph/m2_tf_bn.png')
st.subheader('model 1 nbn wavelet')
st.image('img_graph/m2_wv_nbn.png')
st.subheader('model 1 bn wavelet')
st.image('img_graph/m2_wv_bn.png')
st.dataframe(model2)
st.dataframe(model2wv)
st.subheader('classification reports nobn')
st.image('class_reports/model_1.png')

st.subheader('model 2 bn')
st.code('''
model_tf2_bn= Sequential()
model_tf2_bn.add(Conv1D(128, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf2_bn.add(BatchNormalization())
model_tf2_bn.add(MaxPooling1D(1))
model_tf2_bn.add(Dropout(0.5))
model_tf2_bn.add(Conv1D(32, 1, activation='relu'))
model_tf2_bn.add(BatchNormalization())
model_tf2_bn.add(MaxPooling1D(1))
model_tf2_bn.add(Flatten())
model_tf2_bn.add(Dense(3, activation='softmax'))

model_tf2_bn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf2_bn.summary()
''')


st.subheader('model 2 nbn')
st.code(''' model_tf2 = Sequential()
model_tf2.add(Conv1D(128, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf2.add(MaxPooling1D(1))
model_tf2.add(Dropout(0.5))
model_tf2.add(Conv1D(32, 1, activation='relu'))
model_tf2.add(MaxPooling1D(1))
model_tf2.add(Flatten())
model_tf2.add(Dense(3, activation='softmax'))

model_tf2.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf2.summary()
     
   
''')
model3= pd.DataFrame({'tfidf loss' : 0.1609 , 'tfidf acc': 0.9478,'tfidf bn loss' : 0.1988 , 'tfidf bn acc': 0.9338},index=[0])
model3wv = pd.DataFrame({"wv nbn loss":0.1740 , "wv nbn acc": 0.9409,"wv bn loss":0.2150 , "wv bn acc": 0.9272},index=[0])
st.subheader('model 2 nbn tfidf')
st.image('img_graph/m3_tf_nbn.png')
st.subheader('model 2 bn tfidf')
st.image('img_graph/m3_tf_bn.png')
st.subheader('model 2 nbn wavelet')
st.image('img_graph/m3_wv_nbn.png')
st.subheader('model 2 bn wavelet')
st.image('img_graph/m3_wv_bn.png')
st.dataframe(model3)
st.dataframe(model3wv)

st.subheader('classification reports nobn')
st.image('class_reports/model_2.png')


st.subheader('model 3 bn')
st.code(''' model_tf3 = Sequential()
model_tf3.add(Conv1D(512, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf3.add(BatchNormalization())
model_tf3.add(MaxPooling1D(1))
model_tf3.add(Dropout(0.2))
model_tf3.add(Conv1D(256, 1, activation='relu'))
model_tf3.add(BatchNormalization())
model_tf3.add(MaxPooling1D(1))
model_tf3.add(Dropout(0.2))
model_tf3.add(Conv1D(128, 1, activation='relu'))
model_tf3.add(BatchNormalization())
model_tf3.add(MaxPooling1D(1))
model_tf3.add(Dropout(0.2))
model_tf3.add(Conv1D(64, 1, activation='relu'))
model_tf3.add(BatchNormalization())
model_tf3.add(MaxPooling1D(1))
model_tf3.add(Flatten())
model_tf3.add(Dropout(0.2))
model_tf3.add(Dense(128, activation='relu'))
model_tf3.add(BatchNormalization())
model_tf3.add(Dense(3, activation='softmax'))
model_tf3.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf3.summary()''')

st.subheader('model 3 nbn')
st.code(''' model_tf3_nobn = Sequential()
model_tf3_nobn.add(Conv1D(512, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf3_nobn.add(MaxPooling1D(1))
model_tf3_nobn.add(Dropout(0.2))
model_tf3_nobn.add(Conv1D(256, 1, activation='relu'))
model_tf3_nobn.add(MaxPooling1D(1))
model_tf3_nobn.add(Dropout(0.2))
model_tf3_nobn.add(Conv1D(128, 1, activation='relu'))
model_tf3_nobn.add(MaxPooling1D(1))
model_tf3_nobn.add(Dropout(0.2))
model_tf3_nobn.add(Conv1D(64, 1, activation='relu'))
model_tf3_nobn.add(MaxPooling1D(1))
model_tf3_nobn.add(Flatten())
model_tf3_nobn.add(Dropout(0.2))
model_tf3_nobn.add(Dense(128, activation='relu'))
model_tf3_nobn.add(Dense(3, activation='softmax'))
model_tf3_nobn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf3_nobn.summary()''')

model4= pd.DataFrame({'tfidf nbn loss' : 0.1832 , 'tfidf nbn acc': 0.9434,'tfidf bn loss' : 0.2804 , 'tfidf bn acc': 0.9292},index=[0])
model4wv = pd.DataFrame({"wv nbn loss":0.1967 , "wv nbn acc": 0.9364,"wv bn loss":0.2393, "wv bn acc": 0.9203},index=[0])
st.subheader('model 3 nbn tfidf')
st.image('img_graph/m4_tf_nbn.png')
st.subheader('model 3 bn tfidf')
st.image('img_graph/m4_tf_bn.png')
st.subheader('model 3 nbn wavelet')
st.image('img_graph/m4_wv_nbn.png')
st.subheader('model 3 bn wavelet')
st.image('img_graph/m4_wv_bn.png')
st.dataframe(model4)
st.dataframe(model4wv)
st.subheader('classification reports nobn')
st.image('class_reports/model_3.png')


st.subheader('model 4 bn')
st.code(''' model_tf4 = Sequential()
model_tf4.add(Conv1D(256, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf4.add(BatchNormalization())
model_tf4.add(MaxPooling1D(1))
model_tf4.add(Dropout(0.2))
model_tf4.add(Conv1D(128, 1, activation='relu'))
model_tf4.add(BatchNormalization())
model_tf4.add(MaxPooling1D(1))
model_tf4.add(Dropout(0.2))
model_tf4.add(Conv1D(32, 1, activation='relu'))
model_tf4.add(BatchNormalization())
model_tf4.add(MaxPooling1D(1))
model_tf4.add(Flatten())
model_tf4.add(Dense(3, activation='softmax'))

model_tf4.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf4.summary()

#train model
''')

st.subheader('model 4 nbn')
st.code('''model_tf4_nobn = Sequential()
model_tf4_nobn.add(Conv1D(256, 1, activation='relu', input_shape=(1,X_train.shape[2])))
model_tf4_nobn.add(MaxPooling1D(1))
model_tf4_nobn.add(Dropout(0.2))
model_tf4_nobn.add(Conv1D(128, 1, activation='relu'))
model_tf4_nobn.add(MaxPooling1D(1))
model_tf4_nobn.add(Dropout(0.2))
model_tf4_nobn.add(Conv1D(32, 1, activation='relu'))
model_tf4_nobn.add(MaxPooling1D(1))
model_tf4_nobn.add(Flatten())
model_tf4_nobn.add(Dense(3, activation='softmax'))

model_tf4_nobn.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model_tf4_nobn.summary()
 ''')

model5= pd.DataFrame({'tfidf nbn loss' : 0.1702 , 'tfidf nbn acc': 0.9455,'tfidf bn loss' : 0.2688 , 'tfidf bn acc': 0.9351},index=[0])
model5wv = pd.DataFrame({"wv nbn loss":0.1854 , "wv nbn acc": 0.9362,"wv bn loss":0.2383, "wv bn acc": 0.9195},index=[0])
st.subheader('model 4 nbn tfidf')
st.image('img_graph/m5_tf_nbn.png')
st.subheader('model 4 bn tfidf')
st.image('img_graph/m5_tf_bn.png')
st.subheader('model 4 nbn wavelet')
st.image('img_graph/m5_wv_nbn.png')
st.subheader('model 4 bn wavelet')
st.image('img_graph/m5_wv_bn.png')
st.dataframe(model5)
st.dataframe(model5wv)

st.subheader('classification reports nobn')
st.image('class_reports/model_4.png')




