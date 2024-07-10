from PIL import Image
import streamlit as st

st.set_page_config(page_title="ColTran", layout="centered")

st.write("""
    # Colorization Transformer
    Upload a black and white image (PNG or JPG) and click 'Colorize' to see the magic!
""")

st.markdown("<br>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload image", type=['png', 'jpg'], accept_multiple_files=False)

st.markdown("<br>", unsafe_allow_html=True)

colorize_button_clicked = st.button("Colorize", key="colorize_button", use_container_width=True, disabled=not (uploaded_file is not None))

if colorize_button_clicked:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)