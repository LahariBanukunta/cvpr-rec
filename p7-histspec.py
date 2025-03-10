import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_match(source, reference):
    """Simplified histogram matching using OpenCV LUT mapping."""
    matched_channels = []
    
    for i in range(3):  # Loop through R, G, B channels
        # Compute CDF for source and reference images
        src_hist, _ = np.histogram(source[:, :, i], bins=256, range=(0, 256))
        ref_hist, _ = np.histogram(reference[:, :, i], bins=256, range=(0, 256))

        src_cdf = np.cumsum(src_hist).astype(np.float32)
        ref_cdf = np.cumsum(ref_hist).astype(np.float32)

        src_cdf /= src_cdf[-1]  # Normalize
        ref_cdf /= ref_cdf[-1]  # Normalize

        # Create lookup table (LUT)
        lut = np.interp(src_cdf, ref_cdf, np.arange(256)).astype(np.uint8)

        # Apply LUT to transform source image
        matched_channels.append(cv2.LUT(source[:, :, i], lut))

    # Merge channels and return final image
    return cv2.merge(matched_channels)

def plot_histogram(image, title):
    """Plots histogram for each RGB channel."""
    colors = ('r', 'g', 'b')
    plt.figure(figsize=(8, 4))
    for i, color in enumerate(colors):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color, label=f"{color.upper()} Channel")
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid()

# Load images
source_image = cv2.imread('dog1.jfif')  
reference_image = cv2.imread('dog2.webp')  

# Convert BGR to RGB for matplotlib
source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

# Perform histogram matching
result_image = hist_match(source_image, reference_image)

# Display images
plt.figure(figsize=(10, 10))

plt.subplot(1, 3, 1)
plt.title('Source Image')
plt.imshow(source_image)
plt.axis("off")

plt.subplot(1, 3, 2)
plt.title('Reference Image')
plt.imshow(reference_image)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.title('Matched Image')
plt.imshow(result_image)
plt.axis("off")

plt.show()

# Plot histograms for original and final images
plot_histogram(source_image, "Histogram of Source Image")
plot_histogram(result_image, "Histogram of Matched Image")

plt.show()
