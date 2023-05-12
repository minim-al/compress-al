from time import time

import io

import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_sample_image
from sklearn.utils import shuffle
from PIL import Image
import streamlit as st


RANDOM_STATE = 42  # fixed random state for replicating results
MAX_N_CLUSTERS = 128  # locked to decrease the load on the server (only used if img_n_colors >= MAX_N_CLUSTERS)
SAMPLING_DIVISOR = 5  # the higher the faster, but less accurate


def compress_image(centers, labels, w, h, d):
        """Compress the image from centroids and labels."""
        return centers[labels].reshape(w, h, d)


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown("# compress-al")

img_buffer = st.file_uploader("Upload an image",
                              accept_multiple_files=False,
                              type=["png", "jpg", "jpeg"])

if img_buffer is not None:
    # loading image buffer into an Image object and then an array
    img_name = img_buffer.name
    img_format = img_name[img_name.find(".")+1:].lower()

    if img_format == "jpg":
        img_format = "jpeg"

    img_raw = Image.open(img_buffer)
    img_array = np.array(img_raw)

    # started timer
    t0 = time()

    # preprocessing the image array
    img_array = img_array.astype(np.float64) / 255
    w, h, d = img_array.shape
    img_array = np.reshape(img_array, (w * h, d))
   
    # setting maximum allowed number of colors/clusters
    img_n_colors = np.unique(img_array, axis=0).shape[0]  # number of unique colors
    if img_n_colors < MAX_N_CLUSTERS:
        max_n_clusters = img_n_colors
    else:
        max_n_clusters = MAX_N_CLUSTERS

    # inputting the number of colors/clusters
    n_clusters = st.sidebar.slider("Number of colors",
                                   min_value=1,
                                   max_value=max_n_clusters,
                                   value=32)  # default starting value is 10
    
    # random sample of pixels to fit the model to
    n_samples = (n_clusters + img_array.shape[0]) // SAMPLING_DIVISOR
    img_array_sample = shuffle(img_array,
                               random_state=RANDOM_STATE,
                               n_samples=n_samples)
    
    # creating the model
    kmeans = KMeans(n_clusters=n_clusters,
                    n_init="auto",
                    random_state=RANDOM_STATE)

    # fitting the model on image array sample
    labels = kmeans.fit(img_array_sample)

    # predicting on original image array
    labels = kmeans.predict(img_array)
    
    
    # compressing original image array using fitted centroids and predicted labels
    img_array_compressed = compress_image(kmeans.cluster_centers_,
                                          labels,
                                          w, h, d)

    # creating an Image object out of rescaled compressed image array (original 0-255 RGB color node)
    img_compressed = Image.fromarray((img_array_compressed * 255).astype(np.uint8), 'RGB')

    
    left_column, right_column = st.columns(2)

    # showing original image
    right_column.write(f"Original ({img_n_colors} colors):")
    right_column.image(img_raw)
    
    left_column.write(f"Compressed ({n_clusters} colors):")
    # showing compressed image
    left_column.image(img_compressed, output_format=img_format)

    # ended timer and writing results
    st.write(f"Compressing image done in {time() - t0:0.3f}s.")
    
    # allocating buffer for IO
    buffer = io.BytesIO()
    
    # writing compressed image to memory buffer and reading from it when pressing the download button
    img_compressed.save(buffer, format=img_format)
    left_column.download_button(
            label="Download compressed image",
            data=buffer,
            file_name=f"compressed-{img_name}",
            mime=f"image/{img_format}"
    )
 
