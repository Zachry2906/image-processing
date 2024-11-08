import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from io import BytesIO
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from image_processing import convert_to_gray, convert_to_binary, convert_to_negative, convert_to_smooth, detect_edge, change_brightness, equalization, rotate, flip, contrast, sharpness

# untuk display histogram
def display_histogram(img):
    # Convert to grayscale if the image is in color
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot(hist, color='#4CAF50')
    ax.set_title("Histogram", color='white', fontweight='bold')
    ax.set_xlabel("Pixel Value", color='white')
    ax.set_ylabel("Frequency", color='white')
    ax.tick_params(colors='white')
    ax.set_ylim([0, 50000])
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    return fig


def process_image(img, menu, params=None):
    """Unified function to process both uploaded images and camera frames"""
    # jika tidak ada parameter, maka set params menjadi dictionary kosong
    if params is None:
        params = {}
    
    if menu == 'Grayscale':
        return convert_to_gray(img)
    elif menu == 'Binary':
        # cvtColor merupakan kepanjangan dari Convert Color
        # COLOR_BGR2GRAY digunakan untuk mengubah warna gambar ke grayscale
        # jika panjang dari img.shape lebih dari 2, maka img adalah gambar berwarna (RGB)
        # jika tidak, maka img adalah gambar grayscale
        return convert_to_binary(cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) > 2 else img, params.get('threshold', 128))
    elif menu == 'Negative':
        return convert_to_negative(img)
    elif menu == 'Edge Detection':
        # img_gray digunakan untuk mengubah gambar ke grayscale
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        # GaussianBlur digunakan untuk menghaluskan gambar dengan kernel 3x3
        img_gaussian = cv.GaussianBlur(img_gray, (3,3), 0)
        # detect_edge digunakan untuk mendeteksi tepi gamabar yang menerima dua parameter, gambar yang sudah dihaluskan dan metode yang digunakan
        return detect_edge(img_gaussian, params.get('method', 'Canny'))
    elif menu == 'Smoothing':
        return convert_to_smooth(img, params.get('factor', 5))
    # params.get digunakan untuk mendapatkan nilai dari parameter yang sesuai dengan key yang diberikan
    # jika key tidak ditemukan, maka nilai default yang diberikan adalah 5
    elif menu == 'Brightness':
        return change_brightness(img, params.get('factor', 1.0))
    elif menu == "Equalization":
        # equalization digunakan untuk menyeimbangkan histogram gambar
        # cvtColor digunakan untuk mengubah warna gambar ke grayscale jika panjang dari img.shape lebih dari 2
        return equalization(cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) > 2 else img)
    elif menu == "Rotate":
        return rotate(img, params.get('rotate_degree', 0))
    elif menu == "Flip":
        return flip(img, params.get('arrow', 'Horizontal'))
    elif menu == "Contrast":
        return contrast(img, params.get('factor', 1.0))
    elif menu == "Sharpness":
        return sharpness(img, params.get('factor', 1.0))
    return img


class VideoTransformer:
    def __init__(self, menu, params):
        self.menu = menu
        self.params = params
    
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed_img = process_image(img, self.menu, self.params)
        return processed_img

def display_image_with_histogram(img, title, column):
    with column:
        st.markdown(f"### {title}")
        # Convert to RGB for display if needed
        # cvtColor digunakan untuk mengubah warna gambar ke RGB jika panjang dari img.shape lebih dari 2
        display_img = cv.cvtColor(img, cv.COLOR_BGR2RGB) if len(img.shape) > 2 else img
        st.image(display_img, use_column_width=True)
        # Convert to grayscale for histogram if needed
        # cvtColor digunakan untuk mengubah warna gambar ke grayscale jika panjang dari img.shape lebih dari 2
        hist_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) > 2 else img
        fig = display_histogram(hist_img)
        st.pyplot(fig, clear_figure=True)
# App setup
st.set_page_config(page_title="Aplikasi Pengolahan Citra 2")
st.title("✨ Aplikasi Pengolahan Citra")
st.markdown("Ubah gambar Anda dengan berbagai fitur pengolahan gambar yang disediakan oleh aplikasi ini. 🎨")

# Sidebar setup
st.sidebar.markdown("## 🛠️ Pengolahan citra")
menu = st.sidebar.selectbox("Pilih Fitur Pengolahan Gambar", 
    ["Grayscale", "Binary", "Negative", "Edge Detection", "Smoothing", "Brightness", "Equalization", "Rotate", "Flip", "Contrast", "Sharpness"])

# Parameters setup
# setelah memilih menu, maka akan muncul parameter yang sesuai dengan menu yang dipilih
params = {}
if menu == 'Binary':
    params['threshold'] = st.sidebar.slider("Threshold", value=128, min_value=0, max_value=255)
elif menu == 'Edge Detection':
    params['method'] = st.sidebar.radio("Pilih Metode Konvolusi", ["Canny", "Sobel", "Robert", "Prewit"])
elif menu == 'Smoothing':
    params['factor'] = st.sidebar.slider("Factor", value=5, min_value=1, max_value=19, step=2)
elif menu == 'Brightness':
    params['factor'] = st.sidebar.slider("Brightness Factor", value=1.0, min_value=0.1, max_value=3.0, step=0.1)
elif menu == "Flip":
    params['arrow'] = st.sidebar.radio("Pilih Arah Flip", ["Horizontal", "Vertical", "Both"])
elif menu == "Contrast":
    params['factor'] = st.sidebar.slider('Factor', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
elif menu == "Rotate":
    params['rotate_degree'] = st.sidebar.slider('Degree', min_value=0, max_value=360, value=0, step=1)
elif menu == "Sharpness":
    params['factor'] = st.sidebar.slider('Factor', min_value=0.1, max_value=3.0, value=1.0, step=0.1)

# Tabs
tab1, tab2 = st.tabs(["📤 Unggah Gambar", "📷 Gunakan Kamera"])

with tab1:
    upload_image = st.file_uploader("", type=["jpg", "png", "jpeg", "bmp"])
    
    if upload_image is not None:
        # numpy array digunakan untuk mengubah file yang diupload menjadi array
        file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
        # imdecode digunakan untuk mendekode file_bytes menjadi gambar, 1 digunakan untuk membaca gambar berwarna
        img = cv.imdecode(file_bytes, 1)
        
        col1, col2 = st.columns(2)
        # Display original image and histogram
        display_image_with_histogram(img, "📸 Gambar Asli", col1)
        
        # Process and display modified image and histogram
        processed_img = process_image(img, menu, params)
        processed_img_rgb = cv.cvtColor(processed_img, cv.COLOR_BGR2RGB)
        # processed_img_rgb digunakan untuk mengubah warna dari gambar yang sudah diolah ke RGB
        display_image_with_histogram(processed_img, "🎨 Gambar sudah diedit", col2)
        img_pil = Image.fromarray(processed_img_rgb)
        # img_pil digunakan untuk mengubah array gambar yang sudah diolah menjadi gambar PIL
        img_bytes = BytesIO()
        # img_bytes = BytesIO() digunakan untuk membuat objek BytesIO
        img_pil.save(img_bytes, format='PNG')
        # img_pil.save digunakan untuk menyimpan gambar ke dalam BytesIO
        img_bytes.seek(0)
        # img_bytes.seek(0) digunakan untuk mengatur posisi pointer ke awal
        st.download_button(
            "Download Gambar",
            data=img_bytes,
            file_name="edited_image.png",
            mime="image/png"
        )
    else:
        st.warning("Silahkan upload gambar terlebih dahulu.")

with tab2:
        start_camera = st.button("📷 Start Camera")
        if start_camera:
            webrtc_streamer(key="example", video_transformer_factory=lambda: VideoTransformer(menu, params))
        else:
            st.info("Click the 'Start Camera' button to begin.")

# Footer and style
st.markdown("---")
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob {display: none;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
