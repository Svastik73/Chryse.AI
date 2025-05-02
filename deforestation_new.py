import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from skimage import filters, morphology, segmentation
import os
from pathlib import Path

def load_and_preprocess(image_path):
    img = cv2.imread(image_path)
    #convert to RGB 
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Gausian blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img
def compute_vegetation_indices(img): #ndvi ^_^
    r = img[:,:,0].astype(float)
    g = img[:,:,1].astype(float)
    b = img[:,:,2].astype(float)
    
    epsilon = 1e-8
    
    pseudo_nir = (g * 0.7 + b * 0.3)
    ndvi = (pseudo_nir - r) / (pseudo_nir + r + epsilon)
    ndvi = (ndvi + 1) / 2  # Scale to 0-1
    
    vari = (g - r) / (g + r - b + epsilon)
    vari = np.clip((vari + 1) / 2, 0, 1)     
    
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
    pixels = img.reshape(-1, 3)\
    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(pixels)
    
    # Reshape labels back to image dimensions
    segmented = labels.reshape(img.shape[:2])
    
    clusters_mean_ndvi = []
    for i in range(n_clusters):
        mask = (segmented == i)
        mean_ndvi = np.mean(ndvi[mask])
        clusters_mean_ndvi.append((i, mean_ndvi))
    clusters_mean_ndvi.sort(key=lambda x: x[1], reverse=True)
    
    # Identify likely forest clusters (highest NDVI)
    forest_clusters = [clusters_mean_ndvi[0][0]]
    
    deforested_clusters = [clusters_mean_ndvi[i][0] for i in range(2, min(4, n_clusters))]
    
    # Create masks
    forest_mask = np.isin(segmented, forest_clusters)
    potential_deforestation_mask = np.isin(segmented, deforested_clusters)
    
 #ndvi threshold
    forest_mask = forest_mask & (ndvi > 0.5)
    
 
    forest_dilated = morphology.binary_dilation(forest_mask, morphology.disk(10))
    
 
    recent_deforestation_candidates = forest_dilated & ~forest_mask & potential_deforestation_mask
    
    refined_deforestation = morphology.remove_small_objects(recent_deforestation_candidates, min_size=100)
    refined_deforestation = morphology.binary_closing(refined_deforestation, morphology.disk(3))
    edges = filters.sobel(segmented)
    edge_mask = edges > filters.threshold_otsu(edges)
    
    pattern_deforestation = refined_deforestation & morphology.binary_dilation(edge_mask, morphology.disk(2))
    
    
    deforestation_mask = refined_deforestation
    
    #saving fike>
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
        forest_overlay[forest_mask] = [0, 150, 0]  
        plt.imshow(forest_overlay)
        plt.title('Detected Forest Areas')
        plt.axis('off')
        
        plt.subplot(2, 3, 5)
        deforest_overlay = img.copy()
        deforest_overlay[deforestation_mask] = [255, 0, 0]
        plt.imshow(deforest_overlay)
        plt.title(f'Potential Deforestation\nArea: {np.sum(deforestation_mask)} pixels')
        plt.axis('off')
        
        plt.subplot(2, 3, 6)
        
        combined_overlay = img.copy()
        combined_overlay[forest_mask] = [0, 150, 0] 
        combined_overlay[deforestation_mask] = [255, 0, 0]
        plt.imshow(combined_overlay)
        plt.title('Combined Detection')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'single_image_deforestation_results.png')
        plt.close()
    
    return deforestation_mask, forest_mask

def analyze_deforestation_patterns(deforestation_mask):
  
    labeled_mask, num_features = morphology.label(deforestation_mask, return_num=True)
    
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
    image_path = r"C:\Users\Svastik Kanwar\Documents\Colorize\deforest_image.jpg"  
    output_dir = r"C:\Users\Svastik Kanwar\Documents\Colorize"
    
    deforestation_mask, forest_mask = detect_deforestation_single_image(
        image_path, output_dir=output_dir
    )
    
    stats = analyze_deforestation_patterns(deforestation_mask)
  
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
