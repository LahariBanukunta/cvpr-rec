import cv2
import numpy as np
import matplotlib.pyplot as plt

def hist_match(source, reference):
    """Performs histogram matching for each channel separately."""
    matched_channels = []
    for i in range(3):  # Loop through B, G, R channels
        src_hist, bins = np.histogram(source[:, :, i].flatten(), 256, [0, 256])
        ref_hist, _ = np.histogram(reference[:, :, i].flatten(), 256, [0, 256])

        src_cdf = np.cumsum(src_hist) / src_hist.sum()
        ref_cdf = np.cumsum(ref_hist) / ref_hist.sum()

        mapping = np.interp(src_cdf, ref_cdf, np.arange(256))
        matched_channel = np.interp(source[:, :, i].flatten(), np.arange(256), mapping).reshape(source[:, :, i].shape)
        
        matched_channels.append(matched_channel)

    matched_image = cv2.merge([np.uint8(matched_channels[0]), np.uint8(matched_channels[1]), np.uint8(matched_channels[2])])
    return matched_image

# Load images in color
source_image = cv2.imread('dog1.jfif')  
reference_image = cv2.imread('dog2.webp')  

# Convert from BGR to RGB for correct color display in matplotlib
source_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB)
reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2RGB)

# Perform histogram matching
result_image = hist_match(source_image, reference_image)

# Display results
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.title('Source Image')
plt.imshow(source_image)
plt.axis('off')

plt.subplot(1, 3, 2)
plt.title('Reference Image')
plt.imshow(reference_image)
plt.axis('off')

plt.subplot(1, 3, 3)
plt.title('Matched Image')
plt.imshow(result_image)
plt.axis('off')

plt.tight_layout()
plt.show()
