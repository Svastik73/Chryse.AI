import tensorflow as tf
import numpy as np
import cv2
import tensorflow as tf
load_model = tf.keras.models.load_model

def load_and_preprocess_image(image_path, target_size):
    """Loads and preprocesses the SAR image."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    image = cv2.resize(image, target_size)  # Resize to match model input
    image = image.astype('float32') / 255.0  # Normalize
    image = np.expand_dims(image, axis=-1)  # Add channel dimension (H, W, 1)
    image = np.expand_dims(image, axis=0)  # Add batch dimension (1, H, W, 1)
    return image

def colorize_sar_image(image_path, model_path, output_path):
    """Loads the model and colorizes the SAR image."""
    model = load_model(model_path)
    input_shape = model.input_shape[1:3]  # Get expected height and width
    
    image = load_and_preprocess_image(image_path, input_shape)
    colorized_image = model.predict(image)[0]  # Remove batch dim
    colorized_image = (colorized_image * 255).astype(np.uint8)  # Convert to uint8
    
    # Convert RGB to BGR for OpenCV saving
    if colorized_image.shape[-1] == 3:
        colorized_image = cv2.cvtColor(colorized_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(output_path, colorized_image)
    print(f"Colorized image saved at: {output_path}")

# User input
image_path = r"C:\Users\Svastik Kanwar\Documents\Colorize\ji.png"
output_path = r"C:\Users\Svastik Kanwar\Documents\Colorize\ouy777.png"
model_path = r"C:\Users\Svastik Kanwar\Documents\Colorize\colorization_model_combined.h5"  # Model file

colorize_sar_image(image_path, model_path, output_path)
