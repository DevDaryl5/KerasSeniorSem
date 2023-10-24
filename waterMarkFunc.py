# from PIL import Image
# import numpy as np

# img = Image.open("test1.jpg")
# img_array = np.array(img)
# print(img_array)

# Import image as a matrix
# make small changes in the matrix that will be watermark
# let ryan know the image dimensions
# create binary matrix of a image
# power notes

import pandas as pd
from dfToGrid import *
from tensorflow.keras.datasets import mnist
import numpy as np

(trainX, trainY), (testX, testY) = mnist.load_data()

df = pd.read_csv("mnist_test.csv")
firstImage = dfToList(df)[0]

def diagonalLine(img):
    startpx = 27
    for every28 in img:
        every28[startpx] += 2
        # print(every28[startpx])
        startpx -= 1

def circle(img):
    # Define the center and radius of the circle
    center_x, center_y = 14, 14
    radius = 10  # Adjust the radius as needed

    # Draw the circle by setting the appropriate pixels to 1
    for y in range(28):
        for x in range(28):
            if (x - center_x)**2 + (y - center_y)**2 <= radius**2:
                img[y][x] += 100

def ringCircle(img):
    # Define the center and radii of the two circles
    center_x, center_y = 14, 14
    outer_radius = 10  # Adjust the outer radius as needed
    inner_radius = 8   # Adjust the inner radius as needed

    # Draw the outer circle by setting the appropriate pixels to 1
    for y in range(28):
        for x in range(28):
            distance_to_center = (x - center_x)**2 + (y - center_y)**2
            if outer_radius**2 >= distance_to_center > inner_radius**2:
                img[y][x] += 100

    # listToImage(img)


def rollingAvg(image, window_size):
    """modified to take an entire 2d numpy array and preform changes on it"""
    new_image = []
    for row in image:
        rolling_avg = []
        for i in range(len(row) - window_size + 1):
            window = row[i:i + window_size]
            avg = (sum(window) // window_size)
            rolling_avg.append(avg)

        # this loop adds the values on the old row to the end of the new list to keep the same dimensions
        for i in range(window_size - 1):
            item = row[(len(row) - window_size + i) + 1]
            rolling_avg.append(item)
        new_image.append(np.array(rolling_avg))

    return np.array(new_image)



# for item in firstImage: #run rolling avg for all pixels 
#     rollingAvg(item,3)
    

def fourCorners(img):
    pass

def someMathFunc(img):
    pass