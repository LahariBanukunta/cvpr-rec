import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image in grayscale
image = cv2.imread("doraemon.webp", cv2.IMREAD_GRAYSCALE)
# 1. Image Negative Transformation
negative = 255 - image
# 4. Piecewise Linear Transformation
def piecewise_linear(img):
    r1, s1 = 50, 0
    r2, s2 = 150, 255
    img = img.astype(np.float32)
    # Apply transformation
    img_transformed = np.piecewise(img, 
                                   [img <= r1, (img > r1) & (img <= r2), img > r2],
                                [lambda r: (s1 / r1) * r,
                                    lambda r: ((s2 - s1) / (r2 - r1)) * (r - r1) + s1,
                                    lambda r: ((255 - s2) / (255 - r2)) * (r - r2) + s2])
    return np.uint8(img_transformed)



piecewise_transformed = piecewise_linear(image)
# Display all images
titles = ['Original', 'Negative', 'Piecewise Linear']
images = [image, negative, piecewise_transformed]
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    plt.axis('off')
plt.tight_layout()
plt.show()
