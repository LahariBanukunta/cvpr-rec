import cv2
import numpy as np

def arithmetic_operations(image1_path, image2_path):
    # Load images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error: One or both images could not be loaded.")
        return

    # Resize image2 to match image1's dimensions
    image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    # Perform arithmetic operations
    add_result = cv2.add(image1, image2)
    subtract_result = cv2.subtract(image1, image2)
    multiply_result = cv2.multiply(image1, image2)
    divide_result = cv2.divide(image1.astype(np.float32), image2.astype(np.float32) + 1e-5)  # Avoid division by zero

    # Normalize division result to 0-255 and convert back to uint8
    divide_result = cv2.normalize(divide_result, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Resize images for display
    display_size = (400,400)
    image1_resized = cv2.resize(image1, display_size)
    image2_resized = cv2.resize(image2, display_size)
    add_resized = cv2.resize(add_result, display_size)
    subtract_resized = cv2.resize(subtract_result, display_size)
    multiply_resized = cv2.resize(multiply_result, display_size)
    divide_resized = cv2.resize(divide_result, display_size)

    # Show results
    cv2.imshow('Image 1', image1_resized)
    cv2.imshow('Image 2', image2_resized)
    cv2.imshow('Addition', add_resized)
    cv2.imshow('Subtraction', subtract_resized)
    cv2.imshow('Multiplication', multiply_resized)
    cv2.imshow('Division', divide_resized)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Paths to images
image1_path = 'doraemon1.webp'  # Replace with your first image file
image2_path = 'ocean.jpg'  # Replace with your second image file

# Call function
arithmetic_operations(image1_path, image2_path)
