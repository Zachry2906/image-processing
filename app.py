import cv2 as cv
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
from image_processing import convert_to_gray, convert_to_binary, convert_to_negative, convert_to_smooth, detect_edge, change_brightness, equalization, rotate, flip, contrast, sharpness

# Histogram function with improved styling
def display_histogram(img):
    hist = cv.calcHist([img], [0], None, [256], [0, 256])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist, color='#4CAF50')
    ax.set_title("Histogram", color='white', fontweight='bold')
    ax.set_xlabel("Pixel Value", color='white')
    ax.set_ylabel("Frequency", color='white')
    ax.tick_params(colors='white')
    fig.patch.set_facecolor('none')
    ax.set_facecolor('none')
    st.pyplot(fig)

# App title and description
st.set_page_config(page_title="Aplikasi Pengolahan Citra 1")
st.title("✨ Aplikasi Pengolahan Citra")
st.markdown("Ubah gambar Anda dengan berbagai fitur pengolahan gambar yang disediakan oleh aplikasi ini. 🎨")

# Sidebar styling
st.sidebar.markdown("## 🛠️ Pengolahan citra")

# Tabs for different input types with icons
tab1, tab2 = st.tabs(["📤 Unggah Gambar", "📷 Gunakan Kamera"])

with tab1:
    upload_image = st.file_uploader("", type=["jpg", "png", "jpeg", "bmp"])

    if upload_image is not None:
        file_bytes = np.asarray(bytearray(upload_image.read()), dtype=np.uint8)
        img = cv.imdecode(file_bytes, 1)
        img2 = cv.imdecode(file_bytes, 0)  # Read as grayscale

        menu = st.sidebar.selectbox("Pilih Fitur Pengolahan Gambar", 
        ["Grayscale", "Binary", "Negative", "Edge Detection", "Smoothing", "Brightness", "Equalization", "Rotate", "Flip" , "Contrast", "Sharpness"])

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 📸 Gambar Asli")
            st.image(cv.cvtColor(img, cv.COLOR_BGR2RGB), use_column_width=True)
            display_histogram(cv.cvtColor(img, cv.COLOR_BGR2GRAY))

        with col2:
            st.markdown("### 🎨 Gambar sudah diedit")

            if menu == 'Grayscale':
                img_gray = convert_to_gray(img)
                st.image(img_gray, use_column_width=True)
                display_histogram(img_gray)

            elif menu == 'Binary':
                threshold = st.sidebar.slider("Threshold", value=128, min_value=0, max_value=255)
                img_binary = convert_to_binary(img2, threshold)
                st.image(img_binary, use_column_width=True)
                display_histogram(img_binary)

            elif menu == 'Negative':
                img_negative = convert_to_negative(img)
                st.image(cv.cvtColor(img_negative, cv.COLOR_BGR2RGB), use_column_width=True)
                display_histogram(cv.cvtColor(img_negative, cv.COLOR_BGR2GRAY))

            elif menu == 'Edge Detection':
                method = st.sidebar.radio("Pilih Metode Konvolusi", ["Canny", "Sobel", "Robert", "Prewit"])
                img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
                img_gaussian = cv.GaussianBlur(img_gray, (3,3), 0)
                img_edge = detect_edge(img_gaussian, method)
                st.image(img_edge, use_column_width=True)
                display_histogram(img_edge)

            elif menu == 'Smoothing':
                factor = st.sidebar.slider("Factor", value=5, min_value=1, max_value=19, step=2)
                img_smoothing = convert_to_smooth(img, factor)
                st.image(cv.cvtColor(img_smoothing, cv.COLOR_BGR2RGB), use_column_width=True)
                display_histogram(cv.cvtColor(img_smoothing, cv.COLOR_BGR2GRAY))

            elif menu == 'Brightness':
                factor = st.sidebar.slider("Brightness Factor", value=1.0, min_value=0.1, max_value=3.0, step=0.1)
                img_brightness = change_brightness(img, factor)
                st.image(cv.cvtColor(img_brightness, cv.COLOR_BGR2RGB), use_column_width=True)
                display_histogram(cv.cvtColor(img_brightness, cv.COLOR_BGR2GRAY))

            elif menu == "Equalization":
                img_equalize = equalization(cv.cvtColor(img, cv.COLOR_BGR2GRAY))
                st.image(img_equalize, use_column_width=True)
                display_histogram(img_equalize)

            elif menu == "Rotate":
                rotate_degree = st.sidebar.number_input('Input Derajat Rotasi', min_value=0, max_value=360)
                img_rotate = rotate(img, rotate_degree)
                st.image(cv.cvtColor(np.array(img_rotate), cv.COLOR_BGR2RGB), use_column_width=True)
                display_histogram(cv.cvtColor(np.array(img_rotate), cv.COLOR_BGR2GRAY))

            elif menu == "Flip":
                arrow = st.sidebar.radio("Pilih Arah Flip", ["Horizontal", "Vertical", "Both"])
                img_flip = flip(img, arrow)
                st.image(cv.cvtColor(np.array(img_flip), cv.COLOR_BGR2RGB), use_column_width=True)
                display_histogram(cv.cvtColor(np.array(img_flip), cv.COLOR_BGR2GRAY))
            
            elif menu == "Contrast":
                factor = st.sidebar.slider('Factor', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
                img_contrast = contrast(img, factor)
                st.image(cv.cvtColor(img_contrast, cv.COLOR_BGR2RGB), use_column_width=True)
                display_histogram(cv.cvtColor(img_contrast, cv.COLOR_BGR2GRAY))
            
            elif menu == "Sharpness":
                factor = st.sidebar.slider('Factor', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
                img_sharpness = sharpness(img, factor)
                st.image(cv.cvtColor(img_sharpness, cv.COLOR_BGR2RGB), use_column_width=True)
                display_histogram(cv.cvtColor(img_sharpness, cv.COLOR_BGR2GRAY))

    else:
        st.warning("Silahkan upload gambar terlebih dahulu.")

with tab2:
    menu = st.sidebar.selectbox("Pilih Fitur Pengolahan Gambar Pada Kamera", 
    ["Grayscale", "Binary", "Negative", "Edge Detection", "Smoothing", "Brightness", "Equalization", "Flip", "Contrast", "Sharpness"])

    # Add a start camera button
    start_camera = st.button("📷 Start Camera")

    if start_camera:
        # Initialize camera
        cap = cv.VideoCapture(0)  # Use default camera (index 0)

        # Check if camera opened successfully
        if not cap.isOpened():
            st.error("❌ Error: Unable to open camera. Please check your camera connection.")
        else:
            # Create placeholders for the video feeds
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📹 Gambar Asli")
                original_placeholder = st.empty()
            with col2:
                st.markdown("### 🖼️ Gambar sudah diolah")
                processed_placeholder = st.empty()

            # Add a stop button with improved styling
            stop_button = st.button("🛑 Stop Camera", key="stop_camera")

            # Process frames
            while not stop_button:
                ret, frame = cap.read()
                if not ret:
                    st.error("❌ Failed to capture frame from camera. The camera might be in use by another application.")
                    break

                try:
                    # Display original frame
                    original_placeholder.image(cv.cvtColor(frame, cv.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

                    # Apply selected effect
                    if menu == 'Grayscale':
                        processed_frame = convert_to_gray(frame)
                    elif menu == 'Binary':
                        threshold = st.sidebar.slider("Threshold", value=128, min_value=0, max_value=255)
                        processed_frame = convert_to_binary(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), threshold)
                    elif menu == 'Negative':
                        processed_frame = convert_to_negative(frame)
                    elif menu == 'Edge Detection':
                        method = st.sidebar.radio("Choose Convolution Method", ["Canny", "Sobel", "Robert", "Prewit"])
                        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
                        frame_gaussian = cv.GaussianBlur(frame_gray, (3,3), 0)
                        processed_frame = detect_edge(frame_gaussian, method)
                    elif menu == 'Smoothing':
                        factor = st.sidebar.slider("Factor", value=5, min_value=1, max_value=19, step=2)
                        processed_frame = convert_to_smooth(frame, factor)
                    elif menu == 'Brightness':
                        factor = st.sidebar.slider("Brightness Factor", value=1.0, min_value=0.1, max_value=3.0, step=0.1)
                        processed_frame = change_brightness(frame, factor)
                    elif menu == "Equalization":
                        processed_frame = equalization(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
                    elif menu == "Flip":
                        arrow = st.sidebar.radio("Choose Flip Direction", ["Horizontal", "Vertical", "Both"])
                        processed_frame = flip(frame, arrow)
                    elif menu == "Contrast":
                        factor = st.sidebar.slider('Factor', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
                        processed_frame = contrast(frame, factor)
                    elif menu == "Sharpness":
                        factor = st.sidebar.slider('Factor', min_value=0.1, max_value=3.0, value=1.0, step=0.1)
                        processed_frame = sharpness(frame, factor)
                    else:
                        processed_frame = frame

                    # Display the processed frame
                    processed_placeholder.image(cv.cvtColor(processed_frame, cv.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

                except Exception as e:
                    st.error(f"An error occurred while processing the frame: {str(e)}")
                    break

                # Check if the stop button has been pressed
                if stop_button:
                    break

            # Release the camera
            cap.release()
    else:
        st.info("Klik tombol 'Start Camera' untuk memulai kamera.")

# Footer with improved styling
st.markdown("---")

# Hide Streamlit style
hide_st_style = """
    <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .viewerBadge_container__1QSob {display: none;}
    </style>
"""
st.markdown(hide_st_style, unsafe_allow_html=True)
