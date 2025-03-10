import cv2

# Load the image
image = cv2.imread("doraemon.webp")  

# Resize the image to a specific size (e.g., 400x400 pixels)
resized_image = cv2.resize(image, (400, 400))  # (width, height)

# Display the resized image
cv2.imshow("Resized Image", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
