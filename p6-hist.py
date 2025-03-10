import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image, title="Histogram"):
    """Calculate and return histogram of an image."""
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.grid()

def histogram_equalization(image):
    """Perform histogram equalization on the image."""
    return cv2.equalizeHist(image)

def clahe_equalization(image):
    """Perform CLAHE (Contrast Limited Adaptive Histogram Equalization)."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Load the image in grayscale
image_path = 'doraemon.webp'  # Replace with your image path
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Ensure the image is loaded correctly
if image is None:
    print("Error: Could not load image.")
else:
    # Resize the image for uniformity
    resized_image = cv2.resize(image, (400, 400))

    # Apply Histogram Equalization & CLAHE
    equalized_image = histogram_equalization(resized_image)
    clahe_image = clahe_equalization(resized_image)

    # Display images
    cv2.imshow("Original Resized Image", resized_image)
    cv2.imshow("Histogram Equalized Image", equalized_image)
    cv2.imshow("CLAHE Image", clahe_image)

    # Plot histograms side by side
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    calculate_histogram(resized_image, "Original Image Histogram")

    plt.subplot(1, 3, 2)
    calculate_histogram(equalized_image, "Histogram Equalized")

    plt.subplot(1, 3, 3)
    calculate_histogram(clahe_image, "CLAHE Histogram")

    plt.tight_layout()
    plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()
