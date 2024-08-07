import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Function to read doodles from CSV
def read_doodles_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    points = df[['x', 'y']].values
    return points

# Function to preprocess doodles for visualization
def preprocess_doodles(points, image_size=(500, 500)):
    # Create a blank image
    image = np.zeros(image_size, dtype=np.uint8)
    
    # Draw the points on the image
    for point in points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < image_size[1] and 0 <= y < image_size[0]:
            image[y, x] = 255  # Set pixel value to white
    
    return image

# Function to draw and process doodles
def process_doodles(csv_file):
    # Read doodles from CSV
    points = read_doodles_from_csv(csv_file)
    
    # Preprocess the doodles into an image
    image = preprocess_doodles(points)
    
    # Apply processing steps as in the original code
    preprocessed_image = cv2.GaussianBlur(image, (5, 5), 0)
    adaptive_thresh = cv2.adaptiveThreshold(preprocessed_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    sobel_x = cv2.Sobel(adaptive_thresh, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(adaptive_thresh, cv2.CV_64F, 0, 1, ksize=3)
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    sobel_edges = cv2.normalize(sobel_edges, None, 0, 255, cv2.NORM_MINMAX)
    sobel_edges = np.uint8(sobel_edges)
    canny_edges = cv2.Canny(preprocessed_image, 50, 150)
    combined_edges = cv2.bitwise_or(sobel_edges, canny_edges)
    kernel = np.ones((7, 7), np.uint8)
    dilated_edges = cv2.dilate(combined_edges, kernel, iterations=1)
    closed_edges = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, kernel)
    
    return image, preprocessed_image, adaptive_thresh, closed_edges

# Load and process doodles from CSV
csv_file = 'isolated.csv'
image, preprocessed_image, adaptive_thresh, closed_edges = process_doodles(csv_file)

# Display results
plt.figure(figsize=(20, 8))
plt.subplot(1, 4, 1)
plt.title('Doodles Image')
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.title('Preprocessed Image')
plt.imshow(preprocessed_image, cmap='gray')
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('Adaptive Thresholding')
plt.imshow(adaptive_thresh, cmap='gray')
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('Final Edges')
plt.imshow(closed_edges, cmap='gray')
plt.axis('off')
plt.show()
