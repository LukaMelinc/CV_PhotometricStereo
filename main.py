import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
from mpl_toolkits.mplot3d import Axes3D

"""
circle data - calibration data/radii info for the object
light_directions.txt - set of 20 light directions
"""


# paths to data
circle_data_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/LightProbe-1/circle_data.txt"
light_directions_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/LightProbe-1/light_directions.txt"
image_directory_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/LightProbe-1"

light_directions = np.loadtxt(light_directions_path).T
print(f"Loaded light directions: {light_directions.shape}")

circle_data = np.loadtxt(circle_data_path)
center_x, center_y, radius = circle_data
print(f"circle center: ({center_x}, {center_y}, radius: {radius})")

# LOADING IMAGES
exclude_name = "ref.JPG"
image_files = [f for f in os.listdir(image_directory_path) if f.endswith(('.JPG')) and exclude_name not in f]

images = []
for img_file in image_files:
    img_path = os.path.join(image_directory_path, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        images.append(img)
    else:
        print(f"failed to load image {img_file}")
    images.append(img)

images = np.stack(images, axis=-1)       # shape: [height, width, num_images]
print(f"Loaded images: {images.shape}")

# ESTIMATING NORMAL MAP

h, w, n = images.shape
I = images.reshape(-1, n).T     # shape: [num_images, h*w]

# solve for normal vectors using LS
normals = np.linalg.lstsq(light_directions, I, rcond=None)[0].T     # shape: (h*w, 3)
normals = normals.reshape(h, w, 3)  # reshape to 3d normal map
normals /= np.linalg.norm(normals, axis=2, keepdims=True)

print(f"Computed normal map: {normals.shape}")


# DEPTH MAP
def frankotchellappa(p, q):
    h, w = p.shape
    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    FX, FY = np.meshgrid(fx, fy)

    F = (1j * FX * np.fft.fft2(p) + 1j * FY * np.fft.fft2(q)) / (FX**2 + FY**2 + 1e-6)
    depth = np.real(np.fft.ifft2(F))
    return depth

# Compute p and q from normals
p = normals[:, :, 0] / normals[:, :, 2]
q = normals[:, :, 1] / normals[:, :, 2]

# Recover depth map
depth_map = frankotchellappa(p, q)
print(f"Recovered depth map: {depth_map.shape}")



# VISUALIZATION OF THE RESULTS
# ploting of the depth map
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(w), np.arange(h))
ax.plot_surface(X, Y, depth_map, cmap='viridis', edgecolor='none')
plt.title('Reconstructed Surface')
plt.show()
