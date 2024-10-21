# Enhanced Mini Lightroom


Enhanced Mini Lightroom is a Streamlit-based web application that allows users to perform various image processing operations in real-time. The app provides a user-friendly interface for editing uploaded images or using a live camera feed.

## Features

- Upload images or use a live camera feed
- Various image processing operations:
  - Grayscale conversion
  - Binary conversion
  - Negative conversion
  - Edge Detection
  - Smoothing
  - Brightness adjustment
  - Histogram Equalization
  - Rotation
  - Flip
  - Contrast adjustment
  - Sharpness adjustment
- Histogram display for original and edited images
- Responsive and attractive interface

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/Zachry2906/image-processing.git
   ```

2. Navigate to the project directory:
   ```
   cd image-processing
   ```

3. Create a virtual environment:
   ```
   python -m venv venv
   ```

4. Activate the virtual environment:
   - For Windows:
     ```
     venv\Scripts\activate
     ```
   - For macOS and Linux:
     ```
     source venv/bin/activate
     ```

5. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

2. Open a browser and go to `http://localhost:8501`

3. Choose the "Upload Image" tab to upload an image or "Use Camera" to use a live camera feed.

4. Select the desired image processing operation from the dropdown menu in the sidebar.

5. Adjust parameters if available.

6. View the processed image results in real-time.

## Project Structure

```
enhanced-mini-lightroom/
│
├── app.py                 # Main Streamlit application file
├── image_processing.py    # Module containing image processing functions
├── requirements.txt       # List of dependencies
├── assets/                # Folder containing image assets
│   └── illustration design.svg
└── README.md              # Project documentation
```

## Contributing

Contributions are always welcome! If you'd like to contribute to this project, please follow these steps:

1. Fork the repository
2. Create a new feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.
