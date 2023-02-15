#streamlit application that displays the results of the model
import streamlit as st
import pandas as pd
import numpy as np



st.set_page_config(
    page_title="streamlit_main",
    page_icon="👋",
)



st.title('feature vectorlerin ortalama değerlerinin dağılımı')


st.subheader('X ekseni featurelarımız')
st.subheader('Y ekseni ortalama tfidf ve wavelet değerlerimiz')
st.subheader('wavelet feature distribution')
st.image('img/waveletdist.png')
st.subheader('tfidf featrue distribution')
st.image('img/tfidfdist.png')



st.subheader('x ekseni featurelarımız')
st.subheader('y ekseni non-zero frekansları')

st.subheader('wavelet feature distribution')
st.image('img/waveletdistfreq2.png')

st.subheader('tfidf featrue distribution')
st.image('img/tfidfdistfreq2.png')
