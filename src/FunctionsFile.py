from PIL import Image
import tensorflow as tf
import pandas as pd
import numpy as np
import os


def ExtractData(*FileNames : str) :

    """
    to use extractData Func u need to passe file name as mentioned below
    test_CleanedLabels = ExtractData('test_CleanedLabels')
    """

    AllData =[]
    dir = '../Data'
    

    
    try : 
        
        for FileName in FileNames :
            FileName +=".txt"
            data = []
            with open(os.path.join(dir,FileName), 'r') as f:
                lines = f.readlines()
                for line in lines:
                    data.append(line.split("\n")[0])
                f.close()
            AllData.append(data)
    except Exception as e : 
        print(e)
    
    return AllData


'''#Resize Function
def resizeImages(images, img_size : tuple):
    
    ResizedImages  = []

    for image in images : 
        
        img = Image.open(image)
        ResizedImage = img.resize(img_size,Image.ANTIALIAS)
        ResizedImage = np.array(ResizedImage)
        data = ResizedImage.astype('float32')
        data /= 255
        ResizedImages.append(data)
    
    return np.array(ResizedImages)

def ConvertToCat(data) :
    return np.array(pd.get_dummies(data))'''


def ReshapeData(*arr):

    AllData = []
    for data in arr : 
        data = data.reshape(data.shape[0],-1)
        data = data.astype('float32')
        data /= 255
        AllData.append(data)

    return AllData


def resizeImages(image, img_size):
    w, h = img_size
    image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

    # Check tha amount of padding needed to be done.
    pad_height = h - tf.shape(image)[0]
    pad_width = w - tf.shape(image)[1]

    # Only necessary if you want to do same amount of padding on both sides.
    if pad_height % 2 != 0:
        height = pad_height // 2
        pad_height_top = height + 1
        pad_height_bottom = height
    else:
        pad_height_top = pad_height_bottom = pad_height // 2

    if pad_width % 2 != 0:
        width = pad_width // 2
        pad_width_left = width + 1
        pad_width_right = width
    else:
        pad_width_left = pad_width_right = pad_width // 2

    image = tf.pad(
        image,
        paddings=[
            [pad_height_top, pad_height_bottom],
            [pad_width_left, pad_width_right],
            [0, 0],
        ],
    )

    image = tf.transpose(image, perm=[1, 0, 2])
    image = tf.image.flip_left_right(image)
    return image











