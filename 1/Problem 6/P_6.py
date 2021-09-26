import numpy as np
from PIL import Image, ImageDraw
from numpy import linalg as LA
import os, fnmatch, glob
from sklearn.metrics import confusion_matrix



def read_data(team_name):
    team_image = []
    part1 = glob.glob('Q6_Dataset/Images/' + team_name +  '?.jpg')
    part2 = glob.glob('Q6_Dataset/Images/' + team_name +  '??.jpg')
    addr_files = part1 + part2
    for addr in addr_files:
        tmp_image = Image.open(addr)
        team_image.append(np.asarray(tmp_image))
    return team_image
    

    
class Chelsea_Man_Classifier:
    def __init__(self):
        self.__rgb_blue = np.array([0.0,0.0,255.0])
        self.__rgb_red = np.array([255.0,0.0,0.0])
    def __calc_img_rgb_mean(self,img_set):
        image_mean = []
        for img in img_set:
            image_mean.append(np.mean(img, axis = (0,1)))
        return image_mean
    def __calc_neasrest_team(self, img_mean):
        x = self.__rgb_blue; xx = self.__rgb_red
        chel_dist = LA.norm(img_mean - self.__rgb_blue)
        manu_dist = LA.norm(img_mean - self.__rgb_red)
        if(chel_dist< manu_dist):
            return 'Chelsea'
        else:
            return 'Manunited'
        
    def predict(self, img_set, label = 'undef'):
        prediction = []
        img_set_mean = self.__calc_img_rgb_mean(img_set)
        for img_mean in img_set_mean:
             prediction.append(self.__calc_neasrest_team(img_mean))
        
        cnt = 0
        for i in range(len(prediction)):
             if(label == prediction[i]):
                 cnt = cnt + 1;    
        return prediction, float(cnt)/len(prediction)
    
    
class_names = ['Chelsea', 'Manunited']
chelsea_img = read_data(team_name = 'c');
manu_img = read_data(team_name = 'm')
chel_manu = Chelsea_Man_Classifier()
prediction_chel, acc_chel = chel_manu.predict(img_set = chelsea_img, label = 'Chelsea')
prediction_manu, acc_manu = chel_manu.predict(img_set = manu_img, label = 'Manunited')

#%% **** Confusion Matrix ****
y_test = []
for i in range(len(chelsea_img)):
    y_test.append('Chelsea')
for i in range(len(manu_img)):
     y_test.append('Manunited')
#y_pred = np.vstack((prediction_chel, prediction_manu))
y_pred = prediction_chel + prediction_manu

    
confusion_mtx = confusion_matrix(y_test, y_pred)
print(confusion_mtx)

import matplotlib.pyplot as plt
import numpy as np
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confidence Matrix',
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
    
confd_matrix = np.asarray([[0.71, 0.286], [0.0182, 0.982]])
plot_confusion_matrix(confd_matrix, class_names)
plot_confusion_matrix(confusion_mtx, class_names)
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred, target_names=class_names))
