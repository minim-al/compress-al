import streamlit as st

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown("# compress-al")

st.markdown("Compresses any image to decrease its size using Machine Learning! This is done by intelligenlty lowering the number of colors in an image while keeping it extremely truthful to the original.")

st.markdown("Choose how many colors you want, and let compress-al do the magic!")

st.markdown("## How-to") 

st.markdown("1. Upload your image.")
st.markdown("2. Choose how many colors you want.")
st.markdown("3. Compress!")