import cv2
import numpy as np
import matplotlib.pyplot as plt

def calculate_histogram(image):
    """Calculate and plot histogram of an image."""
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    plt.figure(figsize=(8, 6))
    plt.plot(histogram, color='black')
    plt.title("Histogram of Image")
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])
    plt.grid()
    plt.show()

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
    # Resize the image to a specific size (e.g., 400x400 pixels)
    resized_image = cv2.resize(image, (400, 400))  # (width, height)

    # Step 1: Calculate and plot histogram of the resized image
    calculate_histogram(resized_image)

    # Step 2: Perform Histogram Equalization
    equalized_image = histogram_equalization(resized_image)

    # Step 3: Perform CLAHE
    clahe_image = clahe_equalization(resized_image)

    # Display results
    cv2.imshow("Original Resized Image", resized_image)
    cv2.imshow("Histogram Equalized Image", equalized_image)
    cv2.imshow("CLAHE Image", clahe_image)

    # Wait for user input to close the images
    cv2.waitKey(0)
    cv2.destroyAllWindows()
