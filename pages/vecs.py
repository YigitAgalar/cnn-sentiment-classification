#streamlit application that displays the results of the model
import streamlit as st


st.set_page_config(
    page_title="streamlit_main",
    page_icon="👋",
)


st.subheader('X ekseni featurelarımız')
st.subheader('Y ekseni ortalama tfidf ve wavelet değerlerimiz')


st.subheader('sample vector 1')
st.image('sample_vecs/vec1000.png')



st.subheader('sample vector 2')
st.image('sample_vecs/vec2500.png')


st.subheader('sample vector 3')
st.image('sample_vecs/vec4300.png')


st.subheader('sample vector 4')
st.image('sample_vecs/vec30000.png')


st.subheader('sample vector 5')
st.image('sample_vecs/vec30000.png')