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
df_tfidf_clean = pd.read_csv("data/sentiment_tfidf.csv",index_col=0)

#read pickle file y_test.pkl
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
#read pickle file X_test.pkl
with open('X_test.pkl', 'rb') as f:
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
model = tf.keras.models.load_model('models/tfidfmodel')
df_tfidf_clean = max_score_extraction(df_tfidf_clean,"tfidf_clean")
st.title('Bitcoin hakkında atılan tweetlerin sentiment analizi')

st.subheader('TFIDF Vektörleri ile Sentiment Analizi')
st.write("Oluşan TFIDF matriksinin şekli: ","(",len(df_tfidf_clean),",",X_test.shape[1],")")

st.subheader('Modelin Özet Bilgileri')
st.code(''' 
model = Sequential()
model.add(Conv1D(32, 3, activation='relu', input_shape=(X_train.shape[1],1)))
model.add(MaxPooling1D(3))
model.add(Flatten())
model.add(Dropout(0.8))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

model.summary()''')
st.subheader('Duyguların dağılımı -1 negatif, 0 nötr, 1 pozitif')
st.write(sentiments["sentiment"].value_counts(normalize=True))
st.bar_chart(sentiments["sentiment"].value_counts(normalize=True))
#what are the columns in the dataset
st.subheader('Veri Seti Özellikleri')
st.write(df_tfidf_clean.columns)
#make a barplot

#what are the columns in the dataset
st.subheader('Veri Seti Özellikleri')
st.write(df_tfidf_clean.columns)
#show the dataset
st.write(df_tfidf_clean[["tfidf_neg_clean","tfidf_neu_clean","tfidf_pos_clean","tfidf_clean_max_score","sentiment"]].sample(10))

st.subheader('Modelin Performansı')
#show the model plots from img folder
st.image('img/tfidf_cnn_acc.png')
st.image('img/tfidf_cnn_loss.png')

st.subheader('Evaluation Sonuçları')
st.write(f'Test verisinde loss _{round(0.20052196085453033,2)}_ Test verisinde accuracy _{round(0.9334022998809814,2)}_')
#st.write('Test verisinde loss: ',model.evaluate(X_test,y_test)[0],'Test verisinde accuracy: ',model.evaluate(X_test,y_test)[1])
#select ["tfidf_neg_clear","tfidf_neu_clear","tfidf_post_clear","tfidf_clean_max_score","sentiment"] column from df_tfidf_clean
#select ["tfidf_neg_clear","tfidf_neu_clear","tfidf_post_clear","tfidf_clean_max_score","sentiment"] column from df_tfidf_clean
#show "tfidf_neg_clear","tfidf_neu_clear","tfidf_post_clear","tfidf_clean_max_score","sentiment" from df_tfidf_clean

allscores = pd.read_csv("data/all_results.csv",index_col=0)
st.subheader('Time Series Analysis')
#select row from allscores dataframe that starts with tfct 
#show row from allscores dataframe that starts with tfct
st.dataframe(allscores[allscores.index.str.startswith("tfct")])


st.image('img/tf_lstm_ts_pred.png')
st.image('img/tf_cnn_ts_pred.png')
st.image('img/tf_hyb_ts_pred.png')