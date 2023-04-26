import math

import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.cluster import KMeans

from sklearn.metrics import pairwise_distances_argmin
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle

from PIL import Image

import streamlit as st

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("# compress-al")

img_buffer = st.file_uploader("Upload an image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])

RANDOM_STATE = 42

if img_buffer is not None:
    # loading image buffer into an Image object and then an array
    img_raw = Image.open(img_buffer)
    img_array = np.array(img_raw)

    # preprocessing the image array
    img_array = img_array.astype(np.float64) / 255
    w, h, d = img_array.shape
    img_array = np.reshape(img_array, (w * h, d))
    
    # inputting the number of colors/clusters
    img_n_colors = np.unique(img_array, axis=0).shape[0]  # number of unique colors
    if img_n_colors < 128:
        max_n_clusters = img_n_colors
    else:
        MAX = 128
    N_CLUSTERS = st.sidebar.slider("Number of colors", min_value=1, max_value=MAX)
    
    
    
    min_samples, max_samples = N_CLUSTERS+img_array.shape[0]/4, img_array.shape[0]
    print("min, max:", min_samples, ",", max_samples)
    np.random.seed(RANDOM_STATE)
    img_array_sample = shuffle(img_array, random_state=RANDOM_STATE, n_samples=np.random.randint(min_samples, max_samples))
    
    kmeans = KMeans(n_clusters=N_CLUSTERS, n_init="auto", random_state=RANDOM_STATE)

    labels = kmeans.fit(img_array_sample)

    # predicting on original
    labels = kmeans.predict(img_array)
    print("Original image labels:", labels, sep="\n\t")

    def compress_image(centers, labels, w, h, d):
        """Compress the image from centroids and labels."""
        return centers[labels].reshape(w, h, d)

    st.image(img_raw)
    
    plt.title(f"Quantized image ({N_CLUSTERS} colors, K-Means)")
    
    fig = (px.imshow(compress_image(kmeans.cluster_centers_, labels, w, h, d))
        .update_xaxes(showticklabels = False, visible=False)
        .update_yaxes(showticklabels = False, visible=False)
        .update_layout(
            margin=dict(l=0,r=0,b=0,t=0),
            paper_bgcolor="Black"
            )
    )
    fig.write_image("compressed_image.png", engine="kaleido", width=2560, height=1440)
    img_compressed = Image.open('compressed_image.png')
    
    st.image(img_compressed)
