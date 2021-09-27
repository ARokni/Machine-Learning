import pandas as pd 
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
import numpy as np
import math 
import random
from sklearn import preprocessing
import matplotlib.pyplot as plt





    
def split_data(data, label, test_size):
    msk = np.random.rand(len(data)) < (1-test_size)
    X_train = data[msk]; y_train = label[msk]
    X_test = data[~msk]; y_test = label[~msk]
    return X_train, X_test, y_train, y_test




data = pd.read_csv("random_dataset.csv")
X = data[data.columns[[0,1]]].values
augment_vect = np.ones(len(X)).reshape((len(X), 1))
y = data[data.columns[[2]]].values
# INAROBAYADDDDDDDDDDDDDDD DASTI BEZANIAAA
X = np.concatenate((X, augment_vect), axis=1)
train_data, X_test, train_label, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#train_data, X_test, train_label, y_test = split_data(X, y, test_size=0.2)
#%%
class logistic_reg:
    def __init__(self, step_size = 0.001):
        self.__step_size = step_size
        #self.__max_iter = max_iter
        self.__theta = []
        self.__classes = []
    def __stop_cond(self, tol, cost_prev, cost_new, max_iter, iteration):
        if np.abs(cost_new - cost_prev)<tol:
            return True
        else:
            return False
    def __logistic_fnc(self, theta, data):
        exponent = math.exp(-1*np.matmul(theta, data))
        return 1/(1+exponent)
    def __cost_fnc(self, theta, data, label):
            h = self.__logistic_fnc(theta, data)
            if(label == 1):
                cost = np.log(h)
            else:
                cost = np.log(1-h)
            return cost
    def __grad(self, data, label,theta ):
         h = self.__logistic_fnc(theta, data)
         return (label - h)*data
    def fit_model(self, dataset, labels, initial_param, max_iter = 100, tol = 0.001):
        FIRST_ITER = 0
        cost_prev = 100000 # fake number
        theta = initial_param
        self.__classes = np.unique(labels)
        fig, axs = plt.subplots(len(self.__classes), 1, figsize=(15,15))
        counter = 0
        for c in self.__classes:
            cost = []
            theta = initial_param
            for i in range(max_iter):
                cost_tmp = 0
                theta_batch = np.array([0.0,0.0,0.0])
                for data, label in zip(dataset, labels):
                    if label == c:
                        label = 1
                    else:
                        label = 0
                    theta = theta + self.__step_size*self.__grad(data, label,theta )
                    cost_tmp = cost_tmp + self.__cost_fnc(theta, data, label)
                cost_new = cost_tmp
                cost.append(cost_new)
                if self.__stop_cond(tol, cost_prev, cost_new, max_iter, iteration = i) and i !=FIRST_ITER:
                    print("Yess")
                    break 
                cost_prev = cost_new
            ax = axs[counter]
            ax.plot((cost), linewidth=2, label="Val")
            ax.legend(loc="upper right", fontsize=14) 
            ax.set_xlabel(" Model {}".format(int(c+1)), fontsize=14) 
            ax.set_ylabel("Log Likelihood", fontsize=14)  
            counter = counter + 1
            self.__theta.append(theta)
        plt.show()
        plt.close()
        return self.__theta
    
    
    
  
clf = logistic_reg(step_size = 0.001)
_max_iter = 1000
_tol = 0.001
_initial_param = np.array([-0.01,-0.01,0.0])
theta_classes = clf.fit_model(dataset = train_data, labels = train_label, 
                            initial_param = _initial_param ,max_iter = _max_iter, tol =_tol)              


#%% 
fig = plt.figure()
ax = plt.axes()
cnt = 1
plt_col = ['r','blue', 'g']
for data, label in zip(X, y):
    col = plt_col[int(label)]
    ax.scatter(data[0], data[1],s = 10, color = col)
for theta, col in zip(theta_classes, plt_col):
    if(cnt == 3):
        x = np.linspace(-10, 10, 1000)
    elif(cnt == 1):
        x = np.linspace(-50, 150, 1000)
    else:
         x = np.linspace(-50, 150, 1000)
    ax.plot(x, -1*theta[2]/theta[1]-(theta[0]/theta[1])*x, label="Class{}".format(cnt), color = col);
    cnt = cnt + 1
    plt.legend(loc="upper right", fontsize=14) 
plt.show()
plt.show()
plt.close()

