from sklearn.datasets import load_digits
#import matplotlib.pyplot as plt
import pandas as pd
import skimage
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

images = load_digits()['images']
target = load_digits()['target']

print('image size is',images.shape[1:])

from statistics import mean, median
shape = (len(images), images.shape[1] * images.shape[2])
reshaped_images = images.reshape(shape)


xtrain, xtest, ytrain, ytest = train_test_split(reshaped_images, target, random_state=1, 
                                                stratify=target, test_size=0.2)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, random_state=1, stratify=ytrain,
                                                  test_size=0.6)

C  = [0.1, 1, 10, 30, 50, 70,100] 
gamma =  [0.0001,0.001,0.01,1,10,50,100]


model = SVC(C=C[0], gamma=gamma[0])
model.fit(xtrain, ytrain)
predicted = model.predict(xvalid)
print(type(predicted))



def test_predict_not_oneclass():
    for this_val in predicted:
        result = np.all(predicted == this_val)
        assert result == False



def test_predicted_from_all_class():
    flag = False
    for digit in target:
        flag = True
        if digit not in predicted:
            flag = False
            print()
            print(str(digit) +" is not present in the predicted")
            break
    assert flag == True