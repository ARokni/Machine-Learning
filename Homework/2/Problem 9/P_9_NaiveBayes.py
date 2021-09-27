import pandas as pd 
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import math 
import random
import matplotlib.pyplot as plt
import itertools

random.seed(10)
np.random.seed(seed=9)




    
    
def _confusion_martix_(y_pred, y_test):
    classes = np.unique(y_test)
    for cnt in range(len(classes)):
        indexes = [i for i,x in enumerate(y_test) if x == classes[cnt]]
        for ind in indexes:
            y_test[ind] = cnt
        indexes = [i for i,x in enumerate(y_pred) if x == classes[cnt]]
        for ind in indexes:
            y_pred[ind] = cnt
    y_pred = y_pred.reshape(y_pred.shape[0])
    y_test = y_test.reshape(y_test.shape[0])
    y_test = y_test.astype(int)
    y_pred = y_pred.astype(int)
    conf_mat = np.zeros([len(classes), len(classes)])
    for i in range(len(y_test)):
        conf_mat[y_test[i],y_pred[i]] +=1
    f1_score = []
    for label in np.unique(y_test):
        tp = conf_mat[label, label]
        fp = np.sum(conf_mat[:,label]) - tp
        fn = np.sum(conf_mat[label,:]) - tp
        f1_score.append(tp/(tp + 0.5*(fp+fn)))
        
    return conf_mat, f1_score
        
	
    
    
def split_data(data, label, test_size):
    msk = np.random.rand(len(data)) < (1-test_size)
    X_train = data[msk]; y_train = label[msk]
    X_test = data[~msk]; y_test = label[~msk]
    return X_train, X_test, y_train, y_test

def scale_data(data_set):
    cnt = 0
    for data in data_set.T:
        mean = np.mean(data)
        var = np.var(data)
        data  = (data - mean*np.ones(len(data)))/np.sqrt(var)
        data_set.T[cnt,:] = data
        cnt = cnt + 1
    return data_set

#%%





train_data = pd.read_csv("/home/amiredge/Desktop/MyCourses/Master/Term3 Fall 99 /ML/Hw/HW2/Codes/TinyMNIST/trainData.csv").values
train_label = pd.read_csv("/home/amiredge/Desktop/MyCourses/Master/Term3 Fall 99 /ML/Hw/HW2/Codes/TinyMNIST/trainLabels.csv").values
X_test = pd.read_csv("/home/amiredge/Desktop/MyCourses/Master/Term3 Fall 99 /ML/Hw/HW2/Codes/TinyMNIST/testData.csv").values
y_test = pd.read_csv("/home/amiredge/Desktop/MyCourses/Master/Term3 Fall 99 /ML/Hw/HW2/Codes/TinyMNIST/testLabels.csv").values

#%%  Noisy Moon
#data = pd.read_csv("Noisy Moons.csv")
#X = data[data.columns[[0,1]]].values
#y = data[data.columns[[2]]].values
#train_data, X_test, train_label, y_test = split_data(X, y, test_size=0.2,)
#train_data, X_test, train_label, y_test = split_data(X, y, test_size=0.2)


class gaussian_naive_bayes:
    def __init__(self, smoothness):
        self.__mu = []
        self.__sigma = []
        self.__prior = []
        self.__classes = []
        self.__smoothness = smoothness
    def __calc_prior(self, labels):
        for c in self.__classes:
            class_num = (np.where(labels == c))[0]
            self.__prior.append(len(class_num)/len(labels))
    def train_moments(self, train_data, labels):
        labels = labels.reshape(labels.shape[0])
        self.__classes = set(labels)
        #labels = np.asarray(labels[labels.columns[[0]]] ).reshape(labels.shape[0])
        for c in self.__classes:
            ind = np.where(labels == c)
            data_j = train_data[ind]
            mu_j = np.mean(data_j, axis = 0)
            sigma_j = np.cov(data_j.T)
            self.__sigma.append(np.diag(sigma_j))
            self.__mu.append(mu_j)
        self.__calc_prior(labels)
    def __prob_pdf(self,x, mu, sigma):
        sigma = sigma + self.__smoothness
        mean = mu; stdev = np.sqrt(sigma)
        expo = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2)))) 
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * expo 
    def __calc_prob(self, data_pred):
        prob_classes = []
        for j in range(len(self.__classes)):
            prob = 1
            for i in range(len(data_pred)):
                prob =  prob*self.__prob_pdf(data_pred[i], self.__mu[j][i], self.__sigma[j][i])
            prob_classes.append(self.__prior[j]*prob)
        return prob_classes
            
        
    def predict(self, data_pred):
        pred = []
        return np.argmax(self.__calc_prob(data_pred))
        
    def get_moments(self):
        return self.__mu, self.__sigma, self.__prior

_smoothness = 10**(-4)
nv_clf = gaussian_naive_bayes(smoothness = _smoothness)
nv_clf.train_moments(train_data, train_label)
mu, sigma, prior = nv_clf.get_moments()

label_test = np.copy(y_test)
label_test = label_test.reshape(len(y_test))

cnt = 0
y_prediction = np.zeros(len(X_test))
for i in range(len(X_test)):
    test_data = X_test[i]
    prediction = nv_clf.predict(test_data)
    y_prediction[i] = prediction
    if(prediction == label_test[i]):
        cnt = cnt + 1
print("acc: ", cnt/len(X_test))

#%%
_conf_mat, f1_score = _confusion_martix_(y_prediction, y_test)
print("f1: ",f1_score)

print(_conf_mat)

#%% Use Packages

from sklearn.naive_bayes import GaussianNB   

clf = GaussianNB(var_smoothing= _smoothness)  

# fitting the classifier
clf.fit(train_data, train_label)
y_pred = clf.predict(X_test)

from sklearn.metrics import accuracy_score

print("The accuracy of the model is: %.1f%%" % (accuracy_score(y_test, y_pred)*100))
        


