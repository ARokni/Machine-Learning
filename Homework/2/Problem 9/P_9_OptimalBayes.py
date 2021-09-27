import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from numpy import linalg as LA
import numpy as np
import math 
import random
num_seed = 26
random.seed(num_seed)


    
def split_data(data, label, test_size):
    msk = np.random.rand(len(data)) < (1-test_size)
    X_train = data[msk]; y_train = label[msk]
    X_test = data[~msk]; y_test = label[~msk]
    return X_train, X_test, y_train, y_test

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
        




train_data = pd.read_csv("trainData.csv")
train_label = pd.read_csv("trainLabels.csv")
X_test = pd.read_csv("testData.csv")
y_test = pd.read_csv("testLabels.csv")


#%%  Noisy Moon
#data = pd.read_csv("/content/drive/My Drive/Patt2/Noisy Moons.csv")
#X = data[data.columns[[0,1]]]
#y = data[data.columns[[2]]]

#train_data, X_test, train_label, y_test = train_test_split(X, y, test_size=0.2, random_state=num_seed)


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
        self.__classes = set(labels[labels.columns[0]])
        labels = np.asarray(labels[labels.columns[[0]]] ).reshape(labels.shape[0])
        for c in self.__classes:
            ind = np.where(labels == c)
            data_j = train_data.iloc[ind].values
            mu_j = np.mean(data_j, axis = 0)
            sigma_j = np.cov(data_j.T)
            self.__sigma.append((sigma_j))
            self.__mu.append(mu_j)
        self.__calc_prior(labels)
    def __prob_pdf(self,x, mu, sigma):
        sig_test = np.copy(sigma)
       # sigma = sigma + self.__smoothness*np.identity(len(mu))
        #if(LA.det(sigma)==0):
          #  hh = 25
        #mean = mu; stdev = np.sqrt(sigma)
        expo = math.exp(-0.5*np.matmul(np.matmul((x-mu).T, LA.pinv(sigma)), (x-mu)))
        return (1 / ((math.sqrt(2 * math.pi))**(len(mu)/2) * LA.det(sigma+self.__smoothness*np.identity(len(mu)))**(-0.5))) * expo 
    def __calc_prob(self, data_pred):
        prob_classes = []
        for j in range(len(self.__classes)):
            prob =  self.__prior[j]*self.__prob_pdf(data_pred, self.__mu[j], np.asarray(self.__sigma[j]))
            prob_classes.append(prob)
        return prob_classes
            
        
    def predict(self, data_pred):
        pred = []
        return np.argmax(self.__calc_prob(data_pred))
        
    def get_moments(self):
        return self.__mu, self.__sigma, self.__prior

_smoothness = 10**(0)
nv_clf = gaussian_naive_bayes(smoothness = _smoothness)
nv_clf.train_moments(train_data, train_label)
mu, sigma, prior = nv_clf.get_moments()

label_test = y_test.values
label_test = label_test.reshape(len(y_test))
x_test = X_test.values
y_prediction = np.zeros(len(X_test))
cnt = 0
for i in range(len(x_test)):
    test_data = x_test[i]
    prediction = nv_clf.predict(test_data)
    y_prediction[i] = prediction
    if(prediction == label_test[i]):
        cnt = cnt + 1
print("acc: ", cnt/len(x_test))
conf_mat, f1_score = _confusion_martix_(y_prediction, y_test.values)
print("f1: ",f1_score)

from sklearn.naive_bayes import GaussianNB   

clf = GaussianNB(var_smoothing= _smoothness)  

# fitting the classifier
clf.fit(train_data, train_label)
y_pred = clf.predict(X_test)
_confusion_martix_
from sklearn.metrics import accuracy_score

print("The accuracy of the model is: %.1f%%" % (accuracy_score(y_test, y_pred)*100))
        

label_test = y_test.values
label_test = label_test.reshape(len(y_test))
x_test = X_test.values
y_prediction = np.zeros(len(X_test))
for i in range(len(x_test)):
    test_data = x_test[i]
    prediction = nv_clf.predict(test_data)
    y_prediction[i] = prediction
    if(prediction == label_test[i]):
        cnt = cnt + 1
print("acc: ", cnt/len(x_test))


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
        
conf_mat, f1_score = _confusion_martix_(y_prediction, y_test.values)
print("f1: ",f1_score)
