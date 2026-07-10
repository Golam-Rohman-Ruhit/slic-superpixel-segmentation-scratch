import numpy as np
from sklearn.cluster import KMeans
from skimage import color
from PIL import Image
from scipy.ndimage import zoom
from skimage.filters import gaussian
import os
from matplotlib import pyplot as plt
from copy import deepcopy
from skimage.morphology.binary import binary_erosion


def slic(image_data, k=100, lambda_d=0.5, scale=1.0, gaussian_sigma=1.5):
    # Apply Gaussian filter
    image_data = gaussian(image_data, sigma=gaussian_sigma)
    image_data = zoom(image_data, [scale, scale, 1], order=1, mode='nearest')

    height, width = image_data.shape[0:2]
    lab_map = color.rgb2lab(image_data)
    l_channel = lab_map[:, :, 0]
    a_channel = lab_map[:, :, 1]
    b_channel = lab_map[:, :, 2]

    [x_mesh, y_mesh] = np.meshgrid(np.arange(width), np.arange(height))

    features = [l_channel.reshape(-1),
                a_channel.reshape(-1),
                b_channel.reshape(-1),
                lambda_d * x_mesh.reshape(-1),
                lambda_d * y_mesh.reshape(-1)]
    features = np.transpose(np.array(features), axes=[1, 0])

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(features)
    cluster_centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    label_map = labels.reshape(height, width)
    label_map = zoom(label_map, [1 / scale, 1 / scale], order=0, mode='nearest')

    return label_map, cluster_centers


def show_image_with_superpixel(image_data, label_map, cluster_centers):
    plt.subplot(1, 3, 1)
    plt.imshow(image_data)

    image_margin = deepcopy(image_data)
    all_label_values = np.unique(label_map).tolist()

    for value in all_label_values:
        bwmap = (label_map == value)
        this_value_margin = np.logical_and(bwmap, np.logical_not(binary_erosion(bwmap)))
        image_margin[this_value_margin, 0] = 255
        image_margin[this_value_margin, 1] = 255
        image_margin[this_value_margin, 2] = 255

    plt.subplot(1, 3, 2)
    plt.imshow(image_margin)

    reconstruct_image_lab = np.zeros(image_data.shape)

    for value in np.unique(label_map).tolist():
        this_lab = cluster_centers[value, 0:3]
        reconstruct_image_lab[label_map == value, 0:3] = this_lab

    reconstruct_image = color.lab2rgb(reconstruct_image_lab)

    plt.subplot(1, 3, 3)
    plt.imshow(reconstruct_image)
    plt.show()


# Example usage with image paths
image_paths = ['pic1.avif', 'pic2.avif', 'pic3.avif']
for image_path in image_paths:
    if os.path.exists(image_path):
        image = np.array(Image.open(image_path))
        label_map, cluster_centers = slic(image, k=150, scale=0.5, lambda_d=0.5)
        show_image_with_superpixel(image, label_map, cluster_centers)
    else:
        print(f"Image file {image_path} not found.")
