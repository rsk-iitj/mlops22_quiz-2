from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
import pandas as pd
import skimage
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

images = load_digits()['images']
target = load_digits()['target']

print('image size is',images.shape[1:])

plt.figure(figsize=(10, 10))
for idx, val in enumerate([100, 200, 300, 1500]):
    plt.subplot(2, 2, idx + 1)
    plt.imshow(images[val])
    plt.axis('off')
    plt.title('original')
plt.show()

from statistics import mean, median
shape = (len(images), images.shape[1] * images.shape[2])
reshaped_images = images.reshape(shape)


xtrain, xtest, ytrain, ytest = train_test_split(reshaped_images, target, random_state=1, 
                                                stratify=target, test_size=0.2)
xtrain, xvalid, ytrain, yvalid = train_test_split(xtrain, ytrain, random_state=1, stratify=ytrain,
                                                  test_size=0.6)

xtrain.shape, xvalid.shape, xtest.shape



C  = [0.1, 1, 10, 30, 50, 70,100] 
gamma =  [0.0001,0.001,0.01,1,10,50,100]

C_list = []
gamma_list = []
train_acc_list = []
valid_acc_list = []
test_acc_list = []

mean_row = []
median_row = []
min_row=[]
max_row= []


for c in C:
    for gm in gamma:
        model = SVC(C = c, gamma = gm)
        model.fit(xtrain, ytrain)
        train_acc = model.score(xtrain,ytrain )
        valid_acc = model.score(xvalid, yvalid)
        test_acc = model.score(xtest, ytest)
        
        C_list.append(c)
        gamma_list.append(gm)
        train_acc_list.append(train_acc)
        valid_acc_list.append(valid_acc)
        test_acc_list.append(test_acc)
        min_row.append(min([train_acc,valid_acc,test_acc]))
        median_row.append(median((train_acc,valid_acc,test_acc)))
        max_row.append(max([train_acc,valid_acc,test_acc]))
        mean_row.append(mean([train_acc,valid_acc,test_acc]))


df = pd.DataFrame({'C' : C_list, 'gamma' : gamma_list, 'train_accuracy' : train_acc_list, 
                   'dev_accuracy' : valid_acc_list,'test_accuracy' : valid_acc_list})
display(df)
display(df_row)

df_row = pd.DataFrame({'Mean' : mean_row, 'max' : max_row, 'median' : median_row, 
                   'min' : min_row})

max_score = df['dev_accuracy'].max()
best_params = df[df['dev_accuracy'] == max_score]

best_c = best_params['C'].values[0]
best_gamma = best_params['gamma'].values[0]



model = SVC(C = best_c, gamma = best_gamma)
model.fit(xtrain, ytrain)


train_acc = model.score(xtrain, ytrain)
valid_acc = model.score(xvalid, yvalid)
test_acc = model.score(xtest, ytest)   


df_final = pd.DataFrame({'train_accuracy' : [train_acc],'dev_accuracy' : [valid_acc], 
                         'test_accuracy' : [test_acc]})
df_final.index = ['C ' + str(best_c) + ' gamma ' + str(best_gamma)]

display(df_final)