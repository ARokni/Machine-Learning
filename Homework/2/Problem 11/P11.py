from sklearn.linear_model import LogisticRegression
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
import numpy as np
import math 
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt


data = pd.read_csv("quality_test.csv")
X =  data[data.columns[[0,1]]].values
y =  data[data.columns[[2]]].values

max_deg = 6
from sklearn import datasets



def augment_data(X):
    X_aug = []
    for x in X:
        x_tmp = []
        for i in range(1,max_deg+1):
            for j in range(i+1):
                element_aug = [x[0]**(i-j)*x[1]**(j)]
                x_tmp.append(element_aug)
        X_aug.append(x_tmp)
    X_aug = np.asarray(X_aug)
    X_aug = X_aug.reshape(X_aug.shape[0],X_aug.shape[1])
    return X_aug

X_aug = augment_data(X)
Y = y.reshape(len(y))
logreg = LogisticRegression(random_state=0).fit(X_aug, Y)






x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
h = .02  # step size in the mesh
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
xx_aug = augment_data(np.asarray([xx.ravel(), yy.ravel()]).T)
Z = logreg.predict(xx_aug)
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
plt.figure(1, figsize=(4, 3))
#plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
plt.contour(xx, yy, Z, levels=[0], colors='k')

# Plot also the training points
x_part_I = X[0:56]
x_part_II = X[56:117]
#plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
plt.scatter(x_part_I[:, 0], x_part_I[:, 1], color = 'b',label = 'Class I' )
plt.scatter(x_part_II[:, 0], x_part_II[:, 1], color = 'r', label = 'Class II'  )
plt.legend(loc="upper right", fontsize=14)

plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())

plt.xticks(())
plt.yticks(())

