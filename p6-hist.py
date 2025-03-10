import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the image in grayscale
image = cv2.imread("doraemon.webp", cv2.IMREAD_GRAYSCALE)
# 2. Log Transformation (c * log(1 + r))
c = 255 / np.log(1 + np.max(image))  # Compute constant
log_transformed = c * np.log(1 + image)  # Apply log function
log_transformed = np.uint8(log_transformed)  # Convert back to uint8


# 3. Power Law (Gamma) Transformation (s = c * r^γ)
gamma = 2.2  # Change this value to adjust brightness
gamma_transformed = np.array(255 * (image / 255) ** gamma, dtype=np.uint8)


titles = ['Original',  'Log Transform', 'Power Law (Gamma)']
images = [image, log_transformed, gamma_transformed]
plt.figure(figsize=(10, 6))
for i in range(3):
    plt.subplot(2, 3, i+1)
    plt.imshow(images[i], cmap='gray')
    plt.title(titles[i])
    




plt.show()
