import cv2
import numpy as np
import matplotlib.pyplot as plt

def colorize_sar_image(sar_image_path, output_path=None):
    """
    Colorize a SAR image with Earth-like colors using OpenCV.
    
    Parameters:
    -----------
    sar_image_path : str
        Path to the SAR image file
    output_path : str, optional
        Path to save the colorized image. If None, the image will be displayed
        
    Returns:
    --------
    colorized_img : ndarray
        The colorized SAR image
    """
    # Read the SAR image in grayscale
    sar_img = cv2.imread(sar_image_path, cv2.IMREAD_GRAYSCALE)
    
    if sar_img is None:
        raise FileNotFoundError(f"Could not read image: {sar_image_path}")
    
    # Apply histogram equalization for better contrast
    equalized_img = cv2.equalizeHist(sar_img)
    
    # Create an Earth-like colormap
    # This custom colormap is inspired by natural Earth colors:
    # - Dark blue for water bodies
    # - Green for vegetation
    # - Brown for soil/land
    # - White for snow/ice or urban areas
    
    # First, create a normalized image (0-1 range)
    normalized_img = equalized_img.astype(np.float32) / 255.0
    
    # Create an empty RGB image
    colorized_img = np.zeros((sar_img.shape[0], sar_img.shape[1], 3), dtype=np.uint8)
    
    # Create masks for different intensity ranges
    water_mask = normalized_img < 0.3
    vegetation_mask = (normalized_img >= 0.3) & (normalized_img < 0.5)
    soil_mask = (normalized_img >= 0.5) & (normalized_img < 0.8)
    urban_snow_mask = normalized_img >= 0.8
    
    # Apply colors based on masks
    # Water (dark blue)
    colorized_img[water_mask] = [50, 70, 120]
    
    # Vegetation (green)
    colorized_img[vegetation_mask] = [50, 120, 50]
    
    # Soil/land (brown)
    colorized_img[soil_mask] = [100, 70, 40]
    
    # Urban/snow (white/light gray)
    colorized_img[urban_snow_mask] = [200, 200, 200]
    
    # Alternatively, you can use OpenCV's built-in colormaps:
    # colorized_img = cv2.applyColorMap(equalized_img, cv2.COLORMAP_JET)  # or another colormap
    
    # Add texture and details (optional)
    # Apply slight blur to simulate atmosphere
    colorized_img = cv2.GaussianBlur(colorized_img, (3, 3), 0)
    
    # Enhance details
    sharpening_kernel = np.array([[-0.5, -0.5, -0.5],
                                  [-0.5,  5.0, -0.5],
                                  [-0.5, -0.5, -0.5]])
    colorized_img = cv2.filter2D(colorized_img, -1, sharpening_kernel)
    
    # Save or display the result
    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))
        print(f"Colorized image saved to {output_path}")
    else:
        # Display the result using matplotlib
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.title("Original SAR Image")
        plt.imshow(sar_img, cmap='gray')
        plt.axis('off')
        
        plt.subplot(122)
        plt.title("Colorized SAR Image")
        plt.imshow(cv2.cvtColor(colorized_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return colorized_img

# Example usage
if __name__ == "__main__":
    # Replace with your SAR image path
    sar_image_path = "ji.png"
    
    try:
        # Colorize and display the image
        colorized_img = colorize_sar_image(sar_image_path)
        
        # Optionally save the result
        # colorize_sar_image(sar_image_path, "colorized_sar.png")
    except Exception as e:
        print(f"Error: {e}")