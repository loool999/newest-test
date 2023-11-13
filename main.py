import cv2
import numpy as np

def count_matching_pixels(target_path, custom_path):
    # Read images
    target_img = cv2.imread(target_path)
    custom_img = cv2.imread(custom_path)

    # Convert images to grayscale
    target_gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
    custom_gray = cv2.cvtColor(custom_img, cv2.COLOR_BGR2GRAY)

    # Threshold images to create binary masks
    _, target_thresh = cv2.threshold(target_gray, 127, 255, cv2.THRESH_BINARY)
    _, custom_thresh = cv2.threshold(custom_gray, 127, 255, cv2.THRESH_BINARY)

    # Compute the absolute difference between the binary masks
    diff = cv2.absdiff(target_thresh, custom_thresh)

    # Count non-zero pixels (matching pixels)
    matching_pixels = np.count_nonzero(diff)

    return matching_pixels

# Example usage
target_image_path = 'path/to/target_image.jpg'
custom_image_path = 'path/to/custom_image.jpg'

result = count_matching_pixels(target_image_path, custom_image_path)
print(f"Number of matching pixels: {result}")
