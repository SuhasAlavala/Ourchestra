import numpy as np # We'll be storing our data as numpy arrays
import os # For handling directories
from PIL import Image # For handling the images
import matplotlib.pyplot as plt
import matplotlib.image as mpimg # Plotting

lookup = dict()
reverselookup = dict()
count = 0
for j in os.listdir('leapgestrecog/00/'):
    if not j.startswith('.'): # If running this code locally, this is to
                              # ensure you aren't reading in hidden folders
        lookup[j] = count
        reverselookup[count] = j
        count = count + 1

x_data = []
y_data = []
datacount = 0 # We'll use this to tally how many images are in our dataset
for i in range(0, 10): # Loop over the ten top-level folders
    for j in os.listdir('leapgestrecog/0' + str(i) + '/'):
        if not j.startswith('.'): # Again avoid hidden folders
            count = 0 # To tally images of a given gesture
            for k in os.listdir('leapgestrecog/0' +
                                str(i) + '/' + j + '/'):
                                # Loop over the images
                img = Image.open('leapgestrecog/0' +
                                 str(i) + '/' + j + '/' + k).convert('L')
                                # Read in and convert to grayscale
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr)
                count = count + 1
            y_values = np.full((count, 1), lookup[j])
            y_data.append(y_values)
            datacount = datacount + count
x_data = np.array(x_data, dtype = 'float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1) # Reshape to be the correct size

for i in range(0, 3):
    plt.imshow(x_data[i*200 , :, :], cmap=plt.cm.gray)
    plt.title(reverselookup[y_data[i*200 ,0]])
    plt.show()

import keras
from keras.utils import to_categorical
y_data = to_categorical(y_data)

x_data = x_data.reshape((datacount, 120, 320, 1))
x_data /= 255

import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(x_data, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y_data, pickle_out)
pickle_out.close()