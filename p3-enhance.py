import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('doraemon.webp', cv2.IMREAD_GRAYSCALE)

# Resize the image to a specific size (e.g., 400x400 pixels)
resized_image = cv2.resize(image, (400, 400))  # (width, height)

# Function for Bit Level Slicing
def bit_level_slicing(image, bit_position):
    return (image & (1 << bit_position))

# Function for Intensity Level Slicing
def intensity_level_slicing(image, low, high):
    sliced_image = np.zeros_like(image)
    sliced_image[(image >= low) & (image <= high)] = image[(image >= low) & (image <= high)]
    return sliced_image

# Function for Brightness Adjustment
def adjust_brightness(image, brightness_value):
    return cv2.add(image, np.full_like(image, brightness_value))

# Function for Contrast Adjustment
def adjust_contrast(image, contrast_value):
    return cv2.convertScaleAbs(image, alpha=contrast_value, beta=0)

# Display the images for comparison
def display_images(images, titles):
    plt.figure(figsize=(10, 10))
    for i, (img, title) in enumerate(zip(images, titles)):
        plt.subplot(2, 3, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Processing images using resized image
bit_sliced = bit_level_slicing(resized_image, 4)  # Using bit position 4 as an example
intensity_sliced = intensity_level_slicing(resized_image, 100, 150)
brightness_adjusted = adjust_brightness(resized_image, 50)
contrast_adjusted = adjust_contrast(resized_image, 1.5)

# Display images
images = [resized_image, bit_sliced, intensity_sliced, brightness_adjusted, contrast_adjusted]
titles = ['Resized Original', 'Bit Level Slicing', 'Intensity Level Slicing', 'Brightness Adjusted', 'Contrast Adjusted']
display_images(images, titles)
