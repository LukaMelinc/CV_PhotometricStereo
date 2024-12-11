import matplotlib.pyplot as plt
import numpy as np
import math

image_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/LightProbe-1/Image_01.JPG"

light_direction_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/light_directions.txt"
refined_light_path = r"/Users/lukamelinc/Desktop/Belgija/Computer vision/WPO3/PSData/cat/refined_light.txt"

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
