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

st.subheader('wavelet feature distribution')
st.image('img/waveletdist.png')
st.subheader('tfidf featrue distribution')
st.image('img/tfidfdist.png')