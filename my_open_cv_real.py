import cv2
import numpy as np
import matplotlib.pyplot as plt

def colorize_sar_image(sar_image_path, output_path=None):
    
    sar_img = cv2.imread(sar_image_path, cv2.IMREAD_GRAYSCALE)
    
    if sar_img is None:
        raise FileNotFoundError(f"Could not read image: {sar_image_path}")
    equalized_img = cv2.equalizeHist(sar_img)

    normalized_img = equalized_img.astype(np.float32) / 255.0

    colorized_img = np.zeros((sar_img.shape[0], sar_img.shape[1], 3), dtype=np.uint8)
    
    water_mask = normalized_img < 0.3
    vegetation_mask = (normalized_img >= 0.3) & (normalized_img < 0.5)
    soil_mask = (normalized_img >= 0.5) & (normalized_img < 0.8)
    urban_snow_mask = normalized_img >= 0.8

    colorized_img[water_mask] = [50, 70, 120]

    colorized_img[vegetation_mask] = [50, 120, 50]
    

    colorized_img[soil_mask] = [100, 70, 40]
    

    colorized_img[urban_snow_mask] = [200, 200, 200]

    colorized_img = cv2.GaussianBlur(colorized_img, (3, 3), 0)
    

    sharpening_kernel = np.array([[-0.5, -0.5, -0.5],
                                  [-0.5,  5.0, -0.5],
                                  [-0.5, -0.5, -0.5]])
    colorized_img = cv2.filter2D(colorized_img, -1, sharpening_kernel)
    

    if output_path:
        cv2.imwrite(output_path, cv2.cvtColor(colorized_img, cv2.COLOR_RGB2BGR))
        print(f"Colorized image saved to {output_path}")
    else:
    
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

if __name__ == "__main__":

    sar_image_path = "ji.png"
    
    try:
     
        colorized_img = colorize_sar_image(sar_image_path)
 
    except Exception as e:
        print(f"Error: {e}")
