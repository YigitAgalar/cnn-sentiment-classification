#streamlit application that displays the results of the model
import streamlit as st


st.set_page_config(
    page_title="streamlit_main",
    page_icon="👋",
)


st.subheader('X ekseni featurelarımız')
st.subheader("Y ekseni ortalama chi2'de seçilen tfidf ve wavelet değerlerimiz")


st.subheader('sample vector 1')
st.image('sample_vecs2/vec1000.png')



st.subheader('sample vector 2')
st.image('sample_vecs2/vec2000.png')


st.subheader('sample vector 3')
st.image('sample_vecs2/vec15000.png')


st.subheader('sample vector 4')
st.image('sample_vecs2/vec30000.png')


st.subheader('sample vector 5')
st.image('sample_vecs2/vec40000.png')