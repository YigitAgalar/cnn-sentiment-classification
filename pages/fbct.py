#streamlit application that displays the results of the model
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score


sentiments = pd.read_csv("sentiments.csv")
df_tfidf_clean = pd.read_csv("data/finbert_clean.csv",index_col=0)


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



df_tfidf_clean = max_score_extraction(df_tfidf_clean,"finbert_clean")
st.title('Bitcoin hakkında atılan tweetlerin sentiment analizi')

st.subheader('Finbert Transformer ile Sentiment Analizi')


st.subheader('Duyguların dağılımı -1 negatif, 0 nötr, 1 pozitif')
st.write(sentiments["sentiment"].value_counts(normalize=True))
#make a barplot
st.bar_chart(sentiments["sentiment"].value_counts(normalize=True))
#what are the columns in the dataset
st.subheader('Veri Seti Özellikleri')
st.write(df_tfidf_clean.columns)
#show the dataset
st.write(df_tfidf_clean[["finbert_neg_clean","finbert_neu_clean","finbert_pos_clean","finbert_clean_max_score","sentiment"]].sample(10))

st.subheader('Modelin Performansı')
#show the model performance
st.write('Accuracy Score:',round(accuracy_score(sentiments["sentiment"],df_tfidf_clean["sentiment"]),2))
st.write('Precision Score:',round(precision_score(sentiments["sentiment"],df_tfidf_clean["sentiment"],average='weighted'),2))
st.write('Recall Score:',round(recall_score(sentiments["sentiment"],df_tfidf_clean["sentiment"],average='weighted'),2))
st.write('F1 Score:',round(f1_score(sentiments["sentiment"],df_tfidf_clean["sentiment"],average='weighted'),2))




allscores = pd.read_csv("data/all_results.csv",index_col=0)
st.subheader('Time Series Analysis')
#select row from allscores dataframe that starts with tfct 
#show row from allscores dataframe that starts with tfct
st.dataframe(allscores[allscores.index.str.startswith("fb")])


st.image('img/fbct_lstm_ts_pred.png')
st.image('img/fbct_cnn_ts_pred.png')
st.image('img/fbct_hyb_ts_pred.png')