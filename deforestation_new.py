import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from skimage import filters, morphology, segmentation
import os
from pathlib import Path

def load_and_preprocess(image_path):
    """
    Load and preprocess a satellite image
    
    Parameters:
    -----------
    image_path : str
        Path to the satellite image
        
    Returns:
    --------
    img : numpy.ndarray
        Preprocessed image
    """
    #image load
    img = cv2.imread(image_path)
    
    # Convert to RGB (OpenCV loads as BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Apply Gaussian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    
    return img

def compute_vegetation_indices(img):
    """
    Compute various vegetation indices
    
    Parameters:
    -----------
    img : numpy.ndarray
        Input RGB image
        
    Returns:
    --------
    indices : dict
        Dictionary containing various vegetation indices
    """
    # Extract channels
    r = img[:,:,0].astype(float)
    g = img[:,:,1].astype(float)
    b = img[:,:,2].astype(float)
    
    # Avoid division by zero
    epsilon = 1e-8
    
    # Calculate pseudo-NDVI (approximating NIR with green)
    pseudo_nir = (g * 0.7 + b * 0.3)
    ndvi = (pseudo_nir - r) / (pseudo_nir + r + epsilon)
    ndvi = (ndvi + 1) / 2  # Scale to 0-1
    
    # Calculate Visible Atmospherically Resistant Index (VARI)
    vari = (g - r) / (g + r - b + epsilon)
    vari = np.clip((vari + 1) / 2, 0, 1)  # Scale to 0-1
    
    # Calculate Excess Green Index (ExG)
    exg = 2*g - r - b
    exg = (exg - np.min(exg)) / (np.max(exg) - np.min(exg) + epsilon)  # Normalize to 0-1
    
    return {
        'ndvi': ndvi,
        'vari': vari,
        'exg': exg
    }

def detect_deforestation_single_image(image_path, output_dir=None):
    """
    Detect potential deforestation areas from a single satellite image
    
    Parameters:
    -----------
    image_path : str
        Path to the satellite image
    output_dir : str, optional
        Directory to save output visualizations
        
    Returns:
    --------
    deforestation_mask : numpy.ndarray
        Binary mask where True indicates potential deforestation
    forest_mask : numpy.ndarray
        Binary mask where True indicates forest
    """
    # Load and preprocess image
    img = load_and_preprocess(image_path)
    
    # Compute vegetation indices
    indices = compute_vegetation_indices(img)
    ndvi = indices['ndvi']
    
    # Use k-means clustering to segment the image based on color
    # Reshape image for clustering
    pixels = img.reshape(-1, 3)
    
    # Apply k-means clustering (adjust n_clusters based on your image complexity)
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape labels back to image dimensions
    segmented = labels.reshape(img.shape[:2])
    
    # Analyze each cluster to determine if it's forest, cleared land, or other
    clusters_mean_ndvi = []
    for i in range(n_clusters):
        mask = (segmented == i)
        mean_ndvi = np.mean(ndvi[mask])
        clusters_mean_ndvi.append((i, mean_ndvi))
    
    # Sort clusters by mean NDVI (higher NDVI indicates more vegetation)
    clusters_mean_ndvi.sort(key=lambda x: x[1], reverse=True)
    
    # Identify likely forest clusters (highest NDVI)
    forest_clusters = [clusters_mean_ndvi[0][0]]
    
    # Identify likely deforested/clearing clusters (moderate-low NDVI but not water/urban)
    # Typically the middle clusters
    deforested_clusters = [clusters_mean_ndvi[i][0] for i in range(2, min(4, n_clusters))]
    
    # Create masks
    forest_mask = np.isin(segmented, forest_clusters)
    potential_deforestation_mask = np.isin(segmented, deforested_clusters)
    
    # Refine forest mask using NDVI threshold
    forest_mask = forest_mask & (ndvi > 0.5)
    
    # Identify potential recent deforestation areas
    # Look for areas that appear cleared but have some remaining vegetation signatures
    # or are adjacent to forests
    
    forest_dilated = morphology.binary_dilation(forest_mask, morphology.disk(10))
    
    # Potential deforestation would be in the dilated area but not in the original forest
    # and also match our deforested clusters
    recent_deforestation_candidates = forest_dilated & ~forest_mask & potential_deforestation_mask
    
    # Apply additional filtering to remove small isolated areas
    refined_deforestation = morphology.remove_small_objects(recent_deforestation_candidates, min_size=100)
    refined_deforestation = morphology.binary_closing(refined_deforestation, morphology.disk(3))
    
    # Look for characteristic linear or rectangular patterns (often associated with human deforestation)
    # Apply edge detection
    edges = filters.sobel(segmented)
    edge_mask = edges > filters.threshold_otsu(edges)
    
    # Combine with potential deforestation areas
    pattern_deforestation = refined_deforestation & morphology.binary_dilation(edge_mask, morphology.disk(2))
    
    # Final deforestation mask combines both approaches
    deforestation_mask = refined_deforestation
    
    # Save visualizations if output directory is specified
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Create visualizations
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        plt.imshow(img)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(2, 3, 2)
        plt.imshow(ndvi, cmap='RdYlGn')
        plt.title('Vegetation Index (NDVI)')
        plt.colorbar(fraction=0.046, pad=0.04)
        plt.axis('off')
        
        plt.subplot(2, 3, 3)
        plt.imshow(segmented, cmap='tab20')
        plt.title('K-means Segmentation')
        plt.axis('off')
        
        plt.subplot(2, 3, 4)
        forest_overlay = img.copy()
        forest_overlay[forest_mask] = [0, 150, 0]  # Green for forest
        plt.imshow(forest_overlay)
        plt.title('Detected Forest Areas')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        deforest_overlay = img.copy()
        deforest_overlay[deforestation_mask] = [255, 0, 0]  # Red for deforestation
        plt.imshow(deforest_overlay)
        plt.title(f'Potential Deforestation\nArea: {np.sum(deforestation_mask)} pixels')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        # Combined overlay
        combined_overlay = img.copy()
        combined_overlay[forest_mask] = [0, 150, 0]  # Green for forest
        combined_overlay[deforestation_mask] = [255, 0, 0]  # Red for deforestation
        plt.imshow(combined_overlay)
        plt.title('Combined Detection')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'single_image_deforestation_results.png')
        plt.close()
    
    return deforestation_mask, forest_mask

def analyze_deforestation_patterns(deforestation_mask):
    """
    Analyze patterns in detected deforestation
    
    Parameters:
    -----------
    deforestation_mask : numpy.ndarray
        Binary mask of detected deforestation
        
    Returns:
    --------
    stats : dict
        Dictionary containing statistics about deforestation patterns
    """
    # Label connected regions
    labeled_mask, num_features = morphology.label(deforestation_mask, return_num=True)
    
    # Calculate region properties
    regions = []
    for i in range(1, num_features + 1):
        region = (labeled_mask == i)
        area = np.sum(region)
        regions.append(area)
    
    # Calculate statistics
    stats = {
        'total_deforested_area': np.sum(deforestation_mask),
        'num_deforested_regions': num_features,
        'avg_region_size': np.mean(regions) if regions else 0,
        'max_region_size': np.max(regions) if regions else 0,
        'min_region_size': np.min(regions) if regions else 0
    }
    
    return stats

def main():
    # Example usage
    image_path = r"C:\Users\Svastik Kanwar\Documents\Colorize\deforest_image.jpg"  # Replace with your image path
    
    # Set output directory
    output_dir = r"C:\Users\Svastik Kanwar\Documents\Colorize"
    
    # Detect deforestation
    deforestation_mask, forest_mask = detect_deforestation_single_image(
        image_path, output_dir=output_dir
    )
    
    # Analyze deforestation patterns
    stats = analyze_deforestation_patterns(deforestation_mask)
    
    # Calculate forest coverage percentage
    total_pixels = deforestation_mask.size
    forest_percentage = (np.sum(forest_mask) / total_pixels) * 100
    deforestation_percentage = (stats['total_deforested_area'] / total_pixels) * 100
    
    # Print results
    print("\nSingle-Image Deforestation Analysis Results:")
    print(f"Forest coverage: {forest_percentage:.2f}%")
    print(f"Potential deforestation: {deforestation_percentage:.2f}%")
    print(f"Number of deforested regions: {stats['num_deforested_regions']}")
    print(f"Average region size: {stats['avg_region_size']:.2f} pixels")
    
    print(f"\nResults saved to {output_dir}/single_image_deforestation_results.png")

if __name__ == "__main__":
    main()