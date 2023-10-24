from tensorflow.keras.datasets import mnist
import numpy as np
from PIL import Image
from PIL import ImageOps
from os import makedirs
from os import path
import shutil



def markFunc(image):
    """ takes an image in the form of a numpy array and returns a marked array
    This function is where we will change what watermarking function we use
    """
    #Put in watermark function here
    # for row in image: #run rolling avg for all pixels 
    #     row = rollingAvg(row,5)
    # image = rollingAvg(image, 5)
    return rollingAvg(image, 3)

#Ignore this for the moment, trying to figure out an issue I have currently
def rollingAvg(image, window_size):
    """modified to take an entire 2d numpy array and preform changes on it"""
    new_image = []
    for row in image:
        rolling_avg = []
        for i in range(len(row) - window_size + 1):
            window = row[i:i + window_size]
            avg = (sum(window) // window_size)
            avg = (row[i]*.9 + avg*.1) // 1
            rolling_avg.append(avg)

        # this loop adds the values on the old row to the end of the new list to keep the same dimensions
        for i in range(window_size - 1):
            item = row[(len(row) - window_size + i) + 1]
            rolling_avg.append(item)
        new_image.append(np.array(rolling_avg))

    return np.array(new_image)

def prepData():
    """Grabs the mnist data and preps it for submission to the DL training algorithm.
    It splits the training and testing sets in half and marks one half.  The images are
    turned from a numpy array into a .jpg file in the and stored in the following folders
    "/dataset/train/mark"
    "/dataset/train/unmark"
    "/dataset/test/mark"
    "/dataset/test/unmark"
    """
    #load and shuffle dataset
    (trainX, trainY), (testX, testY) = mnist.load_data()
    shuffled_train = np.random.permutation(trainX)
    shuffled_test = np.random.permutation(testX)
    
    #split train and test set
    midpoint = len(trainX)//2
    trainMark = shuffled_train[:midpoint]
    trainUnmark = shuffled_train[midpoint:]
    
    midpoint = len(testX)//2
    testMark = shuffled_test[:midpoint]
    testUnmark = shuffled_test[midpoint:]
    
    if path.isdir("dataset"):
        shutil.rmtree("dataset")
    #Set up directories, make sure this deletes old data everytime this is done
    dataset_home = 'dataset/'
    subdirs = ['train/', 'test/']
    for subdir in subdirs:
        # create label subdirectories
        labeldirs = ['mark/', 'unmark/']
        for labldir in labeldirs:
            newdir = dataset_home + subdir + labldir
            makedirs(newdir, exist_ok=True)
    
    
    #Iterate through all images, put them into .jpg files in the correct directory
    #Make sure this deletes old data everytime this done
    trainFolder = 'dataset/train/'
    testFolder = 'dataset/test/'
    i = 0
    for image in trainMark:
        # image = markFunc(image)
        ImageOps.grayscale(Image.fromarray(markFunc(image))).save(trainFolder + 'mark/' + 'mark.' + str(i) + '.jpg')
        i += 1
    
    i = 0
    for image in trainUnmark:
        Image.fromarray(image).save(trainFolder + 'unmark/' + 'unmark.' + str(i) + '.jpg')
        i += 1
    
    i = 0
    for image in testMark:
        # image = markFunc(image)
        ImageOps.grayscale(Image.fromarray(markFunc(image))).save(testFolder + 'mark/' + 'mark.' + str(i) + '.jpg')
        i += 1
    
    i = 0
    for image in testUnmark:
        Image.fromarray(image).save(testFolder + 'unmark/' + 'unmark.' + str(i) + '.jpg')
        i += 1
        
prepData()
# if path.isdir("dataset"):
# #     shutil.rmtree("dataset")
# (trainX, trainY), (testX, testY) = mnist.load_data()
# ImageOps.grayscale(Image.fromarray(markFunc(trainX[0]))).show()

