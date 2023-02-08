#streamlit application that displays the results of the model
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score


sentiments = pd.read_csv("sentiments.csv")
df_tfidf_clean = pd.read_csv("data/sentiment_wv.csv",index_col=0)

#read pickle file y_test.pkl
with open('wvy_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
#read pickle file X_test.pkl
with open('Wv_test.pkl', 'rb') as f:
    X_test = pickle.load(f)

@st.cache
# fonksiyonun çalışması için csvlerdeki column isimleriyle eşleşen name değişkeni fonksiyon çağrılırken verildi
def max_score_extraction(df,name):
    for column in df.columns:
        pd.to_numeric(df[column])
    df[f"{name}_max_score"]=df.max(axis=1)
    df[f"{name}_max_score_column"]=df.idxmax(axis=1)
    df["sentiment"] = df.max(axis=1)
    for value in range(0,len(df[f"{name}_max_score_column"])):
        if df[f"{name}_max_score_column"][value] == df.columns[2]:
            pass
        elif df[f"{name}_max_score_column"][value] == df.columns[0]:
            df[f"{name}_max_score"][value] = df[f"{name}_max_score"][value]* -1
        elif df[f"{name}_max_score_column"][value] == df.columns[1]:
            df[f"{name}_max_score"][value] = 0
    for index in range(0,len(df[f"{name}_max_score_column"])):
        if df[f"{name}_max_score"][index] > 0.05:
            df["sentiment"][index]= int(1)
        elif df[f"{name}_max_score"][index] < -0.05:
            df["sentiment"][index]= int(-1)
        elif df[f"{name}_max_score"][index] >= -0.05 and df[f"{name}_max_score"][index] <= 0.05:
            df["sentiment"][index]= int(0)
    df["sentiment"]=pd.to_numeric(df["sentiment"],downcast='integer')
    return df


#load the model
modelwv = tf.keras.models.load_model('models/wvmodel')
df_tfidf_clean = max_score_extraction(df_tfidf_clean,"wv_clean")
st.title('Bitcoin hakkında atılan tweetlerin sentiment analizi')

st.subheader('Wavelet Vektörleri ile Sentiment Analizi')
st.write("Oluşan Wavelet matriksinin şekli: ","(",len(df_tfidf_clean),",",X_test.shape[1],")")

st.subheader('Modelin Özet Bilgileri')
st.code(''' #build model
modelwv = Sequential()
modelwv.add(Conv1D(32, 3, activation='relu', input_shape=(Wv_train.shape[1],1)))
modelwv.add(MaxPooling1D(3))
modelwv.add(Flatten())
modelwv.add(Dropout(0.8))
modelwv.add(Dense(8, activation='relu'))
modelwv.add(Dense(3, activation='softmax'))

modelwv.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

modelwv.summary()
''')
st.subheader('Duyguların dağılımı -1 negatif, 0 nötr, 1 pozitif')
st.write(sentiments["sentiment"].value_counts(normalize=True))
#make a barplot
st.bar_chart(sentiments["sentiment"].value_counts(normalize=True))
#what are the columns in the dataset
st.subheader('Veri Seti Özellikleri')
st.write(df_tfidf_clean.columns)
#show the dataset
st.write(df_tfidf_clean[["wv_neg_clean","wv_neu_clean","wv_pos_clean","wv_clean_max_score","sentiment"]].sample(10))

st.subheader('Modelin Performansı')
#show the model plots from img folder
st.image('img/wv_cnn_acc.png')
st.image('img/wv_cnn_loss.png')

st.subheader('Evaluation Sonuçları')
st.write(f'Test verisinde loss _{round(0.2488,2)}_ Test verisinde accuracy _{round(0.9139,2)}_')
#st.write('Test verisinde loss: ',modelwv.evaluate(X_test,y_test)[0],'Test verisinde accuracy: ',modelwv.evaluate(X_test,y_test)[1])
#select ["tfidf_neg_clear","tfidf_neu_clear","tfidf_post_clear","tfidf_clean_max_score","sentiment"] column from df_tfidf_clean
#select ["tfidf_neg_clear","tfidf_neu_clear","tfidf_post_clear","tfidf_clean_max_score","sentiment"] column from df_tfidf_clean
#show "tfidf_neg_clear","tfidf_neu_clear","tfidf_post_clear","tfidf_clean_max_score","sentiment" from df_tfidf_clean

allscores = pd.read_csv("data/all_results.csv",index_col=0)
st.subheader('Time Series Analysis')
#select row from allscores dataframe that starts with tfct 
#show row from allscores dataframe that starts with tfct
st.dataframe(allscores[allscores.index.str.startswith("wv")])


st.image('img/wv_lstm_ts_pred.png')
st.image('img/wv_cnn_ts_pred.png')
st.image('img/wv_hyb_ts_pred.png')