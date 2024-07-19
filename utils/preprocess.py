import pandas as pd
import cv2
import numpy as np
import torch
from scipy.io import loadmat
import torch.nn as nn
import torchvision.models as models
import numpy as np
import matplotlib.pyplot as plt
import requests
import numpy as np
import cv2

class_mean = loadmat("../figure_class_mean.mat")

def preprocess(file):
    #print(f'file at: {file}')
    im = cv2.imread(file)
# Assuming 'mean_image_file' is the path to the mean image file, and 'im' is the image loaded using OpenCV# adjust this line based on the format of your mean image file
    mean_data = np.transpose(class_mean['mean'], (2, 3, 1, 0)).squeeze()
    #print(mean_data.shape)
    #print(im.shape)

    IMAGE_DIM = 256
    CROPPED_DIM = 227

# Convert image from RGB to BGR format (OpenCV loads images in BGR by default)
# im = cv2.imread('/path/to/your/image.jpg')  # Uncomment if you haven't loaded the image
    im_data = im # Assuming 'im' is in RGB, reverse channels to BGR

# Resize image
    im_data = cv2.resize(im_data, (IMAGE_DIM, IMAGE_DIM), interpolation=cv2.INTER_LINEAR)
# Convert from uint8 to float32 and subtract mean data
    im_data = np.float32(im_data)
    im_data -= mean_data
    crops_data = np.zeros((CROPPED_DIM, CROPPED_DIM, 3, 10), dtype=np.float32)
    indices = [0, IMAGE_DIM - CROPPED_DIM]
    n = 0
    for i in indices:
        for j in indices:
            crops_data[:, :, :, n] = im_data[i:i+CROPPED_DIM, j:j+CROPPED_DIM, :]
            crops_data[:, :, :, n+5] = crops_data[:, :, ::-1, n]  # flip horizontally
            n += 1
    center = indices[1] // 2
    crops_data[:, :, :, 4] = im_data[center:center+CROPPED_DIM, center:center+CROPPED_DIM, :]
    crops_data[:, :, :, 9] = crops_data[:, :, ::-1, 4]  # flip horizontally
    return torch.from_numpy(crops_data).permute(3,2,0,1)



def preprocess_url(file):
    responseImg = requests.get(file)
    if responseImg.status_code == 200:
        img_array = np.frombuffer(responseImg.content, np.uint8)
    # Decode the image using OpenCV
        im = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
# Assuming 'mean_image_file' is the path to the mean image file, and 'im' is the image loaded using OpenCV# adjust this line based on the format of your mean image file
    mean_data = np.transpose(class_mean['mean'], (2, 3, 1, 0)).squeeze()
    #print(mean_data.shape)
    #print(im.shape)

    IMAGE_DIM = 256
    CROPPED_DIM = 227

# Convert image from RGB to BGR format (OpenCV loads images in BGR by default)
# im = cv2.imread('/path/to/your/image.jpg')  # Uncomment if you haven't loaded the image
    im_data = im # Assuming 'im' is in RGB, reverse channels to BGR

# Resize image
    im_data = cv2.resize(im_data, (IMAGE_DIM, IMAGE_DIM), interpolation=cv2.INTER_LINEAR)
# Convert from uint8 to float32 and subtract mean data
    im_data = np.float32(im_data)
    im_data -= mean_data
    crops_data = np.zeros((CROPPED_DIM, CROPPED_DIM, 3, 10), dtype=np.float32)
    indices = [0, IMAGE_DIM - CROPPED_DIM]
    n = 0
    for i in indices:
        for j in indices:
            crops_data[:, :, :, n] = im_data[i:i+CROPPED_DIM, j:j+CROPPED_DIM, :]
            crops_data[:, :, :, n+5] = crops_data[:, :, ::-1, n]  # flip horizontally
            n += 1
    center = indices[1] // 2
    crops_data[:, :, :, 4] = im_data[center:center+CROPPED_DIM, center:center+CROPPED_DIM, :]
    crops_data[:, :, :, 9] = crops_data[:, :, ::-1, 4]  # flip horizontally
    return torch.from_numpy(crops_data).permute(3,2,0,1)