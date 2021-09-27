import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import itertools

np.random.seed(seed=9)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def plot_fnc(axs, cl1, cl2, cl3, xlabel, ylabel):
    axs.scatter(cl1[:,0], cl1[:,1], color = 'r', label = 'setosa')    
    axs.scatter(cl2[:,0], cl2[:,1], color = 'b', label = 'versicolor')    
    axs.scatter(cl3[:,0], cl3[:,1], color = 'g', label = 'virginica')  
    axs.legend(loc = 'upper left')    
Data = pd.read_csv("Iris.csv") 

cnt = 0

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
        #data  = (data - mean*np.ones(len(data)))/np.sqrt(var)
        data  = (data - np.min(data)*np.ones(len(data)))/(np.max(data)-np.min(data))
        data_set.T[cnt,:] = data
        cnt = cnt + 1
    return data_set
    
class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
setosa = Data[Data.Class.isin(['Iris-setosa'])]
versicolor = Data[Data.Class.isin(['Iris-versicolor'])]
virginica = Data[Data.Class.isin(['Iris-virginica'])]

def _confusion_martix_(y_pred, y_test):
    classes = np.unique(y_test)
    for cnt in range(len(classes)):
        indexes = [i for i,x in enumerate(y_test) if x == classes[cnt]]
        for ind in indexes:
            y_test[ind] = cnt
        indexes = [i for i,x in enumerate(y_pred) if x == classes[cnt]]
        for ind in indexes:
            y_pred[ind] = cnt
       
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
        
	


#%%
_size = 20
fig, axs = plt.subplots(4, 4, figsize=(_size,_size))
for i in range(Data.shape[1]-1):
    for j in range(Data.shape[1]-1):
        cl1 = np.asarray(setosa[setosa.columns[[i,j]]])
        cl2 = np.asarray(versicolor[versicolor.columns[[i,j]]])
        cl3 = np.asarray(virginica[virginica.columns[[i,j]]])
        ax = axs[i,j]
        ax.scatter(cl1[:,0], cl1[:,1], color = 'r',  label = 'setosa')    
        ax.scatter(cl2[:,0], cl2[:,1], color = 'b', label = 'versicolor')    
        ax.scatter(cl3[:,0], cl3[:,1], color = 'g', label = 'virginica')  
        ax.legend(loc = 'upper left')
        ax.set_xlabel(Data.columns[i])
        ax.set_ylabel(Data.columns[j])
        cnt = cnt + 1
#%%

def nearest_neighbour(x_train, y_train, x):
    x = x.reshape((1,len(x)))
    d = []
    for center in x_train:
        d.append(LA.norm(x-x_train))
        
    label = y_train[np.argmin(d)]
    return label
    

X = (Data[Data.columns[[0,1,2,3]]]).values
X = scale_data(X)
y = (Data[Data.columns[4]]).values
X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

predicted_class = []
for x in X_test:
     predicted_class.append(nearest_neighbour(X_train, y_train, x))
 
y_test = y_test.tolist()
        
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
y_prediction = neigh.predict(X_test)
y_prediction = y_prediction.tolist()
cnt = 0
for i in range(len(y_test)):
    if( y_prediction[i] == y_test[i]):
        cnt = cnt + 1
print("Accuracy of the model: ", cnt/len(y_prediction))
conf_mat, f1_score = _confusion_martix_(y_prediction, y_test)


print("Conf Mat: ", conf_mat)
print("F1_Score: ",f1_score)
plot_confusion_matrix(conf_mat, class_names)

#%% Using Packages



from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=8)


from sklearn.neighbors import NearestNeighbors

neigh = KNeighborsClassifier(n_neighbors=1)
neigh.fit(X_train, y_train)
y_prediction = neigh.predict(X_test)

print(classification_report(y_test, y_prediction, target_names=class_names))

from sklearn.metrics import confusion_matrix

confusion_mtx = confusion_matrix(y_test, y_prediction)
plot_confusion_matrix(confusion_mtx, class_names)

print(confusion_mtx)
from sklearn.metrics import accuracy_score
from sklearn import  metrics
print("The accuracy of the model is: %.1f%%" % (accuracy_score(y_test, y_prediction)*100))

# Plot ROC Curve
class_unique = np.unique(y_test)
n_classes = len(class_unique)
# Compute ROC curve and ROC area for each class
from sklearn.metrics import roc_curve, auc 

def _roc(axs, _y_test, _y_prediction):
  falseAlarm, hit, threshold = roc_curve(_y_test, _y_prediction)
  roc_auc =  auc(falseAlarm, hit)
  axs.plot(falseAlarm, hit, color='r')
  axs.set_xlabel("FA")
  axs.set_ylabel("Hit")
  print("AUC: ", roc_auc)
cnt = 0  

def plot_roc(y_test, y_prediction):
    fig, axs = plt.subplots(3, 1, figsize=(5,5))
    _y_test = np.copy(y_test)
    _y_prediction = np.copy(y_prediction)
    class_unique = np.unique(y_test)
    counter = 0
    for c in class_unique:
        for i in range(len(y_test)):
            if y_test[i] == c:
                _y_test[i] = 1
            else:
                _y_test[i] = 0
            if y_prediction[i] == c:
                _y_prediction[i] = 1
            else:
                _y_prediction[i] = 0
        _y_prediction = np.asarray(_y_prediction).astype(int)
        _y_test = np.asarray(_y_test).astype(int)
        _roc(axs[counter], _y_test, _y_prediction)
        counter +=1
    plt.show()
    plt.close()
    
plot_roc(y_test, y_prediction)
