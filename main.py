import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
from mpl_toolkits.mplot3d import Axes3D

# EXTRACTING THE MASK
# Load the mask image (assumed to be white object on black background)
mask_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/Objects/Image_07.png"
mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
# Ensure the mask is binary
_, mask_binary = cv2.threshold(mask, 9, 1, cv2.THRESH_BINARY)
# Check the mask
plt.imshow(mask_binary, cmap='gray')
plt.title("Binary Mask")
plt.axis('off')
plt.show()


# LOADING THE DATA
# paths to data
image_directory_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/Objects"
#circle_data_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/LightProbe-1/circle_data.txt"
light_directions_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/light_directions.txt"
light_directions = np.loadtxt(light_directions_path).T
print(f"Loaded light directions: {light_directions.shape}")




# LOADING IMAGES
exclude_name = "ref.JPG"
image_files = [f for f in os.listdir(image_directory_path) if f.endswith(('.png')) and exclude_name not in f]
images = []
for img_file in image_files:
    img_path = os.path.join(image_directory_path, img_file)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None:
        # Apply mask to the loaded image
        masked_img = img * mask_binary
        images.append(masked_img)
    else:
        print(f"Failed to load image {img_file}")

plt.imshow(images[2])
plt.show()
# Convert the list of masked images to a NumPy array
images = np.stack(images, axis=0)  # Shape: (num_images, height, width)
print(f"Masked images shape: {images.shape}")


images = np.stack(images, axis=-1)       # shape: [height, width, num_images]
#images = images.astype(np.float32) / 255.0    # normalizing the image

h, w, n = images.shape
#print(f"Loaded images: {images.shape}")

# ESTIMATING NORMAL MAP




I = images.reshape(-1, n).T     # reshaping to: [num_images, h*w]


print(light_directions.shape)
print(I.shape)
# 20 vrstic po tri stolpce

# solve for normal vectors using LS
# np.linalg.lstsq solves the equation I = S * N
normals = np.linalg.lstsq(light_directions, I, rcond=None)[0].T     # shape: (h*w, 3)

# reshape to 3d normal map
normals = normals.reshape(h, w, 3)  

# normalizing the normals
normals /= np.linalg.norm(normals, axis=2, keepdims=True)
normals[np.isnan(normals)] = 0
normals[np.isinf(normals)] = 0
#normals[:, :, 2] = np.clip(normals[:, :, 2], 1e-6, None)       # clipping z-values to avoid division error

print(f"Computed normal map: {normals.shape}")

# VISUALIZING THE NORMAL MAP
# Visualize the normal map components
"""plt.figure(figsize=(12, 4))

# Normal X (Red channel)
plt.subplot(1, 3, 1)
plt.imshow(normals[:, :, 0], cmap='jet')
plt.title('Normal X (n_x)')
plt.colorbar()

# Normal Y (Green channel)
plt.subplot(1, 3, 2)
plt.imshow(normals[:, :, 1], cmap='jet')
plt.title('Normal Y (n_y)')
plt.colorbar()

# Normal Z (Blue channel)
plt.subplot(1, 3, 3)
plt.imshow(normals[:, :, 2], cmap='jet')
plt.title('Normal Z (n_z)')
plt.colorbar()

plt.suptitle("Normal Map Components")
plt.show()"""

# Normalize normals to [0, 1] for visualization
normals_vis = (normals - normals.min()) / (normals.max() - normals.min())

# Display the normal map as an RGB image
plt.figure(figsize=(6, 6))
plt.imshow(normals_vis)
plt.title("Normal Map (RGB Visualization)")
plt.axis('off')
plt.show()



# DEPTH MAP
def frankotchellappa(p, q):
    h, w = p.shape
    fx = np.fft.fftfreq(w)
    fy = np.fft.fftfreq(h)
    FX, FY = np.meshgrid(fx, fy)

    F = (1j * FX * np.fft.fft2(p) + 1j * FY * np.fft.fft2(q)) / (FX**2 + FY**2 + 1e-6)
    depth = np.real(np.fft.ifft2(F))
    return depth

# Compute p and q from normals -> recovering the depth map from  the normal map
p = normals[:, :, 0] / (normals[:, :, 2] + 1e-6)
q = normals[:, :, 1] / (normals[:, :, 2] + 1e-6)

# Smooth the gradients to reduce noise
p = cv2.GaussianBlur(p, (5, 5), 1)
q = cv2.GaussianBlur(q, (5, 5), 1)


# Recover depth map
depth_map = frankotchellappa(p, q)
print(f"Recovered depth map: {depth_map.shape}")


# PLOTING OBJECT'S DEPTH MAP


# VISUALIZATION OF THE RESULTS
# ploting of the depth map
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(np.arange(w), np.arange(h))
ax.plot_surface(X, Y, depth_map, cmap='viridis', edgecolor='none')
plt.title('Reconstructed Surface')
plt.show()

# 2D Depth Map Visualization
def display_2d_depth_map(depth_map, mask=None, title="2D Depth Representation"):
    # Mask the depth map to keep only the object area (optional)
    if mask is not None:
        depth_map = depth_map * mask  # Apply mask to depth map

    # Normalize depth values to range [0, 1] for visualization
    depth_map_normalized = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map))

    # Display the depth map with a colormap
    plt.figure(figsize=(8, 6))
    plt.imshow(depth_map_normalized, cmap='viridis')  # Use 'viridis' or other colormaps
    plt.title(title)
    plt.colorbar(label="Normalized Depth")  # Add a color bar for depth scale
    plt.axis('off')
    plt.show()

# Display the 2D depth map
display_2d_depth_map(depth_map, mask=mask_binary, title="2D Depth Map with Depth Colors")
