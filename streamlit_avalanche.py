import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
import tempfile
from io import BytesIO
import base64
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Avalanche Risk Detection",
    page_icon="❄️",
    layout="wide"
)

# Page title and description
st.title("❄️ Avalanche Risk Detection System")
st.markdown("""
This tool analyzes terrain images to identify potential avalanche risk factors including:
- Snow coverage
- Steep slopes
- Potential fracture lines
- Snow texture variance
- Layer indicators
""")

# Function to detect avalanche risk (copied from your script)
def detect_avalanche_risk(image_path):
    """
    Analyzes an image for potential avalanche risk indicators
    
    Args:
        image_path (str): Path to the input image
        
    Returns:
        dict: Analysis results with risk factors and overall risk assessment
    """
    if not os.path.exists(image_path):
        return {"error": "Image file not found"}
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    if image is None:
        return {"error": "Unable to read image file"}
    
    # Convert to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert to HSV for snow detection
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Create a copy for visualization
    output_image = image_rgb.copy()
    
    # Snow detection using HSV thresholding
    # Snow typically has high V (brightness) and low S (saturation)
    lower_snow = np.array([0, 0, 180])
    upper_snow = np.array([180, 50, 255])
    snow_mask = cv2.inRange(image_hsv, lower_snow, upper_snow)
    
    # Calculate snow coverage percentage
    snow_coverage = np.sum(snow_mask > 0) / (snow_mask.shape[0] * snow_mask.shape[1])
    
    # Edge detection for terrain features
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Detect lines using Hough transform (potential fracture lines)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=20)
    
    # Analyze slope angle using edge orientation
    slopes = []
    potential_fracture_lines = 0
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Skip nearly vertical lines (trees, poles, etc.)
            if abs(x2 - x1) < 5:
                continue
                
            slope = abs((y2 - y1) / (x2 - x1))
            angle = np.degrees(np.arctan(slope))
            
            # Steep slopes (30-45 degrees) are higher risk
            if 30 <= angle <= 45:
                cv2.line(output_image, (x1, y1), (x2, y2), (255, 0, 0), 2)
                slopes.append(angle)
                potential_fracture_lines += 1
    
    # Texture analysis for snow patterns
    gray_normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    
    # GLCM features or similar would be better, but using a simplified approach
    texture_blocks = []
    block_size = 50
    for i in range(0, gray.shape[0] - block_size, block_size):
        for j in range(0, gray.shape[1] - block_size, block_size):
            block = gray_normalized[i:i+block_size, j:j+block_size]
            texture_blocks.append(np.std(block))
    
    # Variance in texture indicates potential instability
    texture_variance = np.std(texture_blocks) if texture_blocks else 0
    
    # Color segmentation for layering detection
    pixels = image_rgb.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(pixels)
    labels = kmeans.labels_
    
    # Count transitions between segments as potential layer boundaries
    segmented = labels.reshape(image_rgb.shape[0], image_rgb.shape[1])
    h_transitions = np.sum(np.diff(segmented, axis=1) != 0)
    v_transitions = np.sum(np.diff(segmented, axis=0) != 0)
    layer_indicator = (h_transitions + v_transitions) / (segmented.shape[0] * segmented.shape[1])
    
    # Calculate overall risk score (simplified model)
    risk_factors = {
        "snow_coverage": snow_coverage,
        "steep_slope_count": len(slopes),
        "avg_slope_angle": np.mean(slopes) if slopes else 0,
        "potential_fracture_lines": potential_fracture_lines,
        "texture_variance": texture_variance,
        "layering_indicator": layer_indicator
    }
    
    # Weighted risk calculation
    risk_score = (
        snow_coverage * 0.2 +
        min(len(slopes) / 10, 1.0) * 0.3 +
        (np.mean(slopes) / 45 if slopes else 0) * 0.2 +
        min(texture_variance / 50, 1.0) * 0.15 +
        min(layer_indicator * 10, 1.0) * 0.15
    )
    
    # Risk classification
    risk_level = "Low"
    if risk_score > 0.7:
        risk_level = "High"
    elif risk_score > 0.4:
        risk_level = "Medium"
    
    results = {
        "risk_factors": risk_factors,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "output_image": output_image,
        "snow_mask": snow_mask,
        "edges": edges
    }
    
    return results

# Function to plot results
def get_visualization_plot(results):
    """
    Visualizes the avalanche detection results and returns the figure
    
    Args:
        results (dict): The detection results
    
    Returns:
        plt.figure: Matplotlib figure with the visualization
    """
    if "error" in results:
        return None
    
    fig = plt.figure(figsize=(12, 8))
    
    plt.subplot(1, 2, 1)
    plt.imshow(results["output_image"])
    plt.title("Analyzed Image (Blue lines = steep slopes)")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    # Create risk factor visualization
    factors = results["risk_factors"]
    names = list(factors.keys())
    values = list(factors.values())
    
    # Normalize values between 0 and 1 for visualization
    normalized_values = []
    for i, val in enumerate(values):
        if names[i] == "avg_slope_angle":
            normalized_values.append(min(val / 45, 1.0))
        elif names[i] == "potential_fracture_lines":
            normalized_values.append(min(val / 10, 1.0))
        elif names[i] == "texture_variance":
            normalized_values.append(min(val / 50, 1.0))
        else:
            normalized_values.append(min(val, 1.0))
    
    plt.bar(names, normalized_values)
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha="right")
    plt.title(f"Risk Factors (Overall: {results['risk_level']}, Score: {results['risk_score']:.2f})")
    
    plt.tight_layout()
    return fig

# Function to save uploaded image to temp file
def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        return tmp_file.name

# Function to convert matplotlib figure to image
def fig_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return Image.open(buf)

# Create sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Avalanche Risk Detection", "About", "Settings", "Help"])

# Main content based on selected page
if page == "Avalanche Risk Detection":
    # File uploader
    uploaded_file = st.file_uploader("Upload a terrain image", type=["jpg", "jpeg", "png"])
    
    # Analysis options
    with st.expander("Analysis Options"):
        col1, col2 = st.columns(2)
        
        with col1:
            analysis_mode = st.radio("Analysis Mode", ["Standard", "Detailed"])
        
        with col2:
            show_intermediate = st.checkbox("Show intermediate processing steps", value=False)
    
    # Process image button
    process_btn = st.button("Analyze Avalanche Risk", type="primary")
    
    if uploaded_file is not None:
        # Display uploaded image
        st.subheader("Uploaded Image")
        st.image(uploaded_file, width=400)
        
        # Process when button is clicked
        if process_btn:
            with st.spinner("Analyzing image..."):
                # Save uploaded file to temp location
                img_path = save_uploaded_file(uploaded_file)
                
                # Detect avalanche risk
                results = detect_avalanche_risk(img_path)
                
                # Check for errors
                if "error" in results:
                    st.error(f"Error: {results['error']}")
                else:
                    # Create tabs for results
                    tab1, tab2, tab3 = st.tabs(["Analysis Results", "Risk Factors", "Technical Details"])
                    
                    with tab1:
                        # Display risk level with color coding
                        risk_color = {
                            "Low": "green",
                            "Medium": "orange", 
                            "High": "red"
                        }
                        
                        st.subheader("Risk Assessment")
                        st.markdown(f"<h1 style='color:{risk_color[results['risk_level']]}'>{results['risk_level']} Risk</h1>", unsafe_allow_html=True)
                        st.markdown(f"Risk Score: {results['risk_score']:.2f} out of 1.0")
                        
                        # Display visualization
                        fig = get_visualization_plot(results)
                        st.pyplot(fig)
                    
                    with tab2:
                        st.subheader("Detailed Risk Factors")
                        
                        # Create columns for risk factors
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Snow Coverage", f"{results['risk_factors']['snow_coverage']:.2%}")
                            st.metric("Number of Steep Slopes", f"{results['risk_factors']['steep_slope_count']}")
                            st.metric("Average Slope Angle", f"{results['risk_factors']['avg_slope_angle']:.1f}°")
                        
                        with col2:
                            st.metric("Potential Fracture Lines", f"{results['risk_factors']['potential_fracture_lines']}")
                            st.metric("Texture Variance", f"{results['risk_factors']['texture_variance']:.2f}")
                            st.metric("Layering Indicator", f"{results['risk_factors']['layering_indicator']:.4f}")
                    
                    with tab3:
                        if show_intermediate and analysis_mode == "Detailed":
                            st.subheader("Intermediate Processing Results")
                            
                            # Show intermediate processing results
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Snow Detection Mask")
                                st.image(results["snow_mask"], width=400)
                            
                            with col2:
                                st.subheader("Edge Detection")
                                st.image(results["edges"], width=400)
                
                # Clean up temp file
                os.unlink(img_path)

elif page == "About":
    st.header("About Avalanche Risk Detection System")
    
    st.markdown("""
    This application uses computer vision and machine learning techniques to analyze terrain images 
    and identify potential avalanche risk factors.
    
    ### How it works:
    
    1. **Snow Detection**: Identifies snow-covered areas using color thresholding
    2. **Slope Analysis**: Detects steep slopes that are prone to avalanches (30-45 degrees)
    3. **Fracture Line Detection**: Locates potential fracture lines using edge detection
    4. **Snow Texture Analysis**: Examines snow texture patterns for instability indicators
    5. **Layer Detection**: Identifies potential weak layers in the snowpack
    
    ### Risk Levels:
    
    - **Low**: Minimal avalanche risk factors detected
    - **Medium**: Some risk factors present, caution advised
    - **High**: Multiple risk factors detected, significant avalanche danger
    
    ### Disclaimer:
    
    This tool is for educational and research purposes only. Always consult official avalanche forecasts 
    and professional guidance before making decisions in avalanche terrain.
    """)

elif page == "Settings":
    st.header("Settings")
    
    st.subheader("Algorithm Parameters")
    
    # Create sections for different parameter groups
    with st.expander("Snow Detection Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Snow Brightness Threshold", min_value=100, max_value=255, value=180)
        with col2:
            st.slider("Snow Saturation Threshold", min_value=0, max_value=100, value=50)
    
    with st.expander("Slope Analysis Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Min Slope Angle (degrees)", min_value=20, max_value=40, value=30)
        with col2:
            st.slider("Max Slope Angle (degrees)", min_value=35, max_value=60, value=45)
    
    with st.expander("Edge Detection Parameters"):
        col1, col2 = st.columns(2)
        with col1:
            st.slider("Edge Lower Threshold", min_value=0, max_value=200, value=50)
        with col2:
            st.slider("Edge Upper Threshold", min_value=100, max_value=300, value=150)
    
    # Save settings button
    st.button("Save Settings")
    
    st.info("Note: Settings are for demonstration purposes in this prototype and don't affect the analysis yet.")

elif page == "Help":
    st.header("Help & Documentation")
    
    st.markdown("""
    ### Uploading Images
    
    1. Click on "Browse files" in the file uploader
    2. Select an image of terrain (ideally a mountain slope with snow)
    3. Click "Analyze Avalanche Risk" to process the image
    
    ### Understanding Results
    
    - **Risk Level**: Overall assessment (Low, Medium, High)
    - **Risk Score**: Numerical score from 0.0 to 1.0
    - **Analyzed Image**: Shows detected steep slopes with blue lines
    - **Risk Factors**: Bar chart showing normalized values for each risk factor
    
    ### Troubleshooting
    
    - If the analysis fails, try a different image with clearer terrain features
    - Images with good contrast and lighting work best
    - For best results, use images taken during daylight hours
    
    ### Contact
    
    For technical support or to report issues, please contact:
    avalanche-detection@example.com
    """)

# Add footer
st.sidebar.markdown("---")
st.sidebar.info("Avalanche Risk Detection v1.0")