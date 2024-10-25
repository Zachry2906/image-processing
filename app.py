import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from image_processing import convert_to_gray, convert_to_binary, convert_to_negative, convert_to_smooth, detect_edge, change_brightness, equalization, rotate, flip, contrast, sharpness

# untuk display histogram
def display_histogram(img):
    # calcHist digunakan untuk menghitung histogram dari gambar
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist, color='#4CAF50')
    ax.set_title("Histogram", color='white', fontweight='bold')
    ax.set_xlabel("Pixel Value", color='white')
    ax.set_ylabel("Frequency", color='white')
    ax.tick_params(colors='white')
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
        st.pyplot(fig)

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
        display_image_with_histogram(processed_img, "🎨 Gambar sudah diedit", col2)
    else:
        st.warning("Silahkan upload gambar terlebih dahulu.")

with tab2:
    start_camera = st.button("📷 Start Camera")
    
    if start_camera:
        cap = cv.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("❌ Error: Unable to open camera. Please check your camera connection.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📹 Gambar Asli")
                original_image = st.empty()
                original_hist = st.empty()
            
            with col2:
                st.markdown("### 🖼️ Gambar sudah diolah")
                processed_image = st.empty()
                processed_hist = st.empty()
            
            stop_button = st.button("🛑 Stop Camera", key="stop_camera")
            
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame from camera.")
                    break
                
                try:
                    # Display original frame and histogram
                    original_image.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                    original_hist.pyplot(display_histogram(cv.cvtColor(frame, cv.COLOR_BGR2GRAY)))
                    
                    # Process frame and display with histogram
                    processed_frame = process_image(frame, menu, params)
                    processed_image.image(cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB) if len(processed_frame.shape) > 2 else processed_frame, 
                                       channels="RGB", use_column_width=True)
                    processed_hist.pyplot(display_histogram(cv.cvtColor(processed_frame, cv.COLOR_BGR2GRAY) if len(processed_frame.shape) > 2 else processed_frame))
                    
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    break
                
                if stop_button:
                    break
            
            cap.release()
    else:
        st.info("Klik tombol 'Start Camera' untuk memulai kamera.")

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
