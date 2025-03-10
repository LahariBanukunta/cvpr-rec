import cv2
import numpy as np

def logical_operations(image1_path, image2_path):
    # Load images in color
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    if image1 is None or image2 is None:
        print("Error: One or both images could not be loaded.")
        return

    # Resize both images to 200x200 for consistency
    image1 = cv2.resize(image1, (400, 400))
    image2 = cv2.resize(image2, (400, 400))

    # Split channels (B, G, R)
    b1, g1, r1 = cv2.split(image1)
    b2, g2, r2 = cv2.split(image2)

    # Perform logical operations on each channel separately
    and_result = cv2.merge([cv2.bitwise_and(b1, b2), cv2.bitwise_and(g1, g2), cv2.bitwise_and(r1, r2)])
    or_result = cv2.merge([cv2.bitwise_or(b1, b2), cv2.bitwise_or(g1, g2), cv2.bitwise_or(r1, r2)])
    xor_result = cv2.merge([cv2.bitwise_xor(b1, b2), cv2.bitwise_xor(g1, g2), cv2.bitwise_xor(r1, r2)])
    not_result1 = cv2.merge([cv2.bitwise_not(b1), cv2.bitwise_not(g1), cv2.bitwise_not(r1)])
    not_result2 = cv2.merge([cv2.bitwise_not(b2), cv2.bitwise_not(g2), cv2.bitwise_not(r2)])

    # Show results
    cv2.imshow('Image 1 (Resized)', image1)
    cv2.imshow('Image 2 (Resized)', image2)
    cv2.imshow('AND Operation', and_result)
    cv2.imshow('OR Operation', or_result)
    cv2.imshow('XOR Operation', xor_result)
    cv2.imshow('NOT Image 1', not_result1)
    cv2.imshow('NOT Image 2', not_result2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
image1_path = 'ocean.jpg'     # Replace with your first image file
image2_path = 'doraemon.webp'  # Replace with your second image file
logical_operations(image1_path, image2_path)
