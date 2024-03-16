import numpy as np
import cv2
from sklearn.cluster import KMeans

print("Image Segmentation Python Script")
print("Please ensure input image is named `input.jpg`\n")

# Enter Number of colors in Segmented Image
dominant_colors = int(input("Enter number of Dominant Colors :: "))
im = cv2.imread("input.jpg")

h, w, _ = im.shape
original_shape = im.shape

# Flatten the image into a 2D array of pixels
all_pixels = im.reshape((-1, 3))

# KMeans clustering
km = KMeans(n_clusters=dominant_colors, n_init=10)  # Set n_init explicitly
km.fit(all_pixels)
centers = km.cluster_centers_

centers = np.array(centers, dtype='uint8')

# Create a new image with clustered colors
new_img = np.zeros((w * h, 3), dtype='uint8')

# Iterate over the image
for ix in range(min(new_img.shape[0], len(km.labels_))):
    new_img[ix] = centers[km.labels_[ix]]

new_img = new_img.reshape((original_shape))

cv2.imwrite('segmented_output.jpg', new_img)

image1 = im
image2 = new_img

# Resize the images to ensure correct dimensions
image1 = cv2.resize(image1, (w, h))
image2 = cv2.resize(image2, (w, h))

abs_diff = np.abs(image1.astype(np.float32) - image2.astype(np.float32))

# Compute mean absolute error (MAE) or root mean squared error (RMSE)
mae = np.mean(abs_diff)
rmse = np.sqrt(np.mean(abs_diff ** 2))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Saving MAE and RMSE to errors.txt
with open('errors.txt', 'w') as f:
    f.write(f"Mean Absolute Error (MAE): {mae}\n")
    f.write(f"Root Mean Squared Error (RMSE): {rmse}\n")
