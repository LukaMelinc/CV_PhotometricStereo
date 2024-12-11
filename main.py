import matplotlib.pyplot as plt
import numpy as np
import math
import os

# paths to data
circle_data_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/LightProbe-1/circle_data.txt"
light_directions = np.loadtxt("/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/LightProbe-1/light_directions.txt")
image_directory = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/LightProbe-1"

# loading the images

def load_images(image_directory):
    images = []
    filenames = sorted(os.listdir(image_directory))
    for filename in filenames:
        img = plt.imread(os.path.join(image_directory, filename))
        if img.ndim == 3:
            img = np.mean(img, axis=2)
        images.append(img)
    return np.array(images)
images = load_images(image_directory)

print(len(images))

assert images.shape[0] == light_directions.shape[0],

def compute_normals(images, light_directions):
    h, w = images[0].shape
    normals = np.zeros((h, w, 3))
    for i in range(h):
        for j in range(w):
            I = images[:, i, j]
            S = light_directions
            # solving the system I = S * n
            n_tilda = np.linalg.lstsq(S, I, rcond=None)[0]  #least squares
            rho = np.linalg.norm(n_tilda)
            normals[i, j] = n_tilda / rho if rho != 0 else [0, 0, 0]
    return normals

normals = compute_normals(images, light_directions)


#visualziation of the noramls
def visualize_noramls(normals):
    normal_map = (normals + 1) / 2
    plt.imshow(normal_map)
    plt.title("Normal Map")
    plt.show()

visualize_noramls(normals)

# surface reconstruction
def reconstruction_surface(normals):
    h, w, _ = normals.shape
    depth_map = np.zeros((h, w))
    for i in range(1, h):
        depth_map[i, 0] = depth_map[i-1, 0] + normals[i, 0, 1] / normals[i, 0, 2]
    for j in range(1, w):
        depth_map[:, j] = depth_map[:, j-1] + normals[:, j, 0] / normals[:, j, 2]
    return depth_map
depth_map = reconstruction_surface(normals)


plt.imshow(depth_map, cmap='gray')
plt.title("depth map")
plt.colorbar()
plt.show()



"""


for i in os.listdir(image_directory):
    with open(os.path.join())

def cosineLaw(angle, radiance):
    Intensity = math.cos(angle) * radiance
    return Intensity

def f_point(p, q):
    f = (2 * p) / (1 + math.sqrt(1 + p**2 + q**2))
    return f

def g_point(p, q):
    g = (2 * q) / (1 + math.sqrt(1 + p*+2 + q*+2))
    return g

# plane z = 1 is the pq plane
def theta(p_source, q_source, p, q):
    cos_theta = (p * p_source + q * q_source + 1)/(math.sqrt(p*+2 + q**2 + 1) * math.sqrt(p_source**2 + q_source**2 + 1))
    theta = math.acos(p * p_source + q * q_source + 1)/(math.sqrt(p*+2 + q**2 + 1) * math.sqrt(p_source**2 + q_source**2 + 1))
    return theta, cos_theta


# matrix multiplication part
I = np.zeros((3, 1))
S = np.zeros((3, 3))
n = np.zeros((3, 1))
"""