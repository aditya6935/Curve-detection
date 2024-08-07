import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to adjust brightness and contrast
def adjust_brightness_contrast(image, brightness=0, contrast=30):
    image = np.int16(image)
    image = image * (contrast / 127 + 1) - contrast + brightness
    image = np.clip(image, 0, 255)
    return np.uint8(image)

# Function to apply CLAHE for contrast enhancement
def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(image)

# Function to preprocess color image to enhance detection
def preprocess_image(image):
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE for contrast enhancement
    equalized_gray_image = apply_clahe(gray_image)

    # Apply Gaussian blur to smooth the image
    blurred_gray_image = cv2.GaussianBlur(equalized_gray_image, (5, 5), 0)

    return blurred_gray_image

# Load the image
input_file = 'occlusion2_rec.png'
image = cv2.imread(input_file)

# Check if the image was successfully read
if image is None:
    print(f"Error: Could not open or find the image '{input_file}'.")
    exit()

# Preprocess the image
preprocessed_image = preprocess_image(image)

# Apply adaptive thresholding to improve edge detection
adaptive_thresh = cv2.adaptiveThreshold(preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

# Apply Sobel operator to detect edges
sobel_x = cv2.Sobel(adaptive_thresh, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(adaptive_thresh, cv2.CV_64F, 0, 1, ksize=3)
sobel_edges = cv2.magnitude(sobel_x, sobel_y)
sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)
sobel_edges = np.uint8(sobel_edges)

# Apply Canny edge detection
canny_edges = cv2.Canny(preprocessed_image, 50, 150)

# Combine Sobel and Canny edges
combined_edges = cv2.bitwise_or(sobel_edges, canny_edges)

# Apply morphological dilation and closing
kernel = np.ones((7, 7), np.uint8)
dilated_edges = cv2.dilate(combined_edges, kernel, iterations=1)
closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)

# Detect circles using Hough Circle Transform
circles = cv2.HoughCircles(closed_edges, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
                           param1=50, param2=30, minRadius=10, maxRadius=100)

# Draw detected circles on the original image
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        cv2.circle(image, center, radius, (0, 255, 0), 2)
        cv2.circle(image, center, 2, (0, 0, 255), 3)

# Display results
plt.figure(figsize=(20, 8))

# Display original image with detected circles
plt.subplot(1, 5, 1)
plt.title('Original Image with Circles')
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# Display preprocessed image
plt.subplot(1, 5, 2)
plt.title('Preprocessed Image')
plt.imshow(preprocessed_image, cmap='gray')
plt.axis('off')

# Display adaptive thresholding result
plt.subplot(1, 5, 3)
plt.title('Adaptive Thresholding')
plt.imshow(adaptive_thresh, cmap='gray')
plt.axis('off')

# Display final edges
plt.subplot(1, 5, 4)
plt.title('Final Edges')
plt.imshow(closed_edges, cmap='gray')
plt.axis('off')

plt.show()
