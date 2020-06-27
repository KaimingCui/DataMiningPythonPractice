#!/usr/bin/env python
# coding: utf-8

# In[38]:

#Questions:
# a. Convert this dataset into numeric by converting y to 1, n to -1 and ? to 0.
# b. Break the dataset into 4 folds with approximately similar ratio of the classes (republi- can/democratic) in each fold. Use 1 fold for parameter tuning only. For the remaining three folds, report average 3-fold classification accuracy (along with standard deviation) for Linear SVM with soft-margin classifier.
# c. Now use SVM with Gaussian Kernel for the same task.



# %load /Users/kaimingcui/Desktop/Semester3/573DataMining/assign4_iuKaimingCui/SVM.py
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
import sys

filename = "house-votes-84.data";
f = open(filename)
outf = open('SVMresult.txt','w')
rows = 435
cols = 16

i = 0;
points = list()
labels = list()
for line in f.readlines():
  point = list();
  line = line.strip('\n')
  data = line.split(',')
  for i in range(len(data)):
    if(i == 0):
      labels.append(data[i])
    else:
      if(data[i] == 'y'):
        data[i] = 1;
      elif(data[i] == 'n'):
        data[i] = -1;
      else:
        data[i] = 0;
      point.append(data[i])
  array = np.array(point)
  points.append(array)

data = np.array(points)

rec = 0;
dec = 0;
for i in labels:
  if(i == "republican"):
    rec += 1;
  else:
    dec += 1;
# num_of_republican 168
# num_of_democrat 267
fold1 = list();
fold2 = list();
fold3 = list();
fold4 = list();

fold1_label = list();
fold2_label = list();
fold3_label = list();
fold4_label = list();

fold_re_c = 0;
fold_de_c = 0;
for i in range(len(labels)):
  if(labels[i] == 'republican'):
    fold_re_c += 1;
    if(fold_re_c <= 42):
      fold1.append(data[i]);
      fold1_label.append(labels[i])
    elif(fold_re_c <= 84):
      fold2.append(data[i]);
      fold2_label.append(labels[i])
    elif(fold_re_c <= 126):
      fold3.append(data[i]);
      fold3_label.append(labels[i])
    else:
      fold4.append(data[i]);
      fold4_label.append(labels[i])
  else:
    fold_de_c += 1;
    if(fold_de_c <= 66):
      fold1.append(data[i]);
      fold1_label.append(labels[i])
    elif(fold_de_c <= 133):
      fold2.append(data[i]);
      fold2_label.append(labels[i])
    elif(fold_de_c <= 200):
      fold3.append(data[i]);
      fold3_label.append(labels[i])
    else:
      fold4.append(data[i]);
      fold4_label.append(labels[i])
fold1 = np.array(fold1);
fold2 = np.array(fold2);
fold3 = np.array(fold3);
fold4 = np.array(fold4);

fold1_label = np.array(fold1_label);
fold2_label = np.array(fold2_label);
fold3_label = np.array(fold3_label);
fold4_label = np.array(fold4_label);

print('Part1:',file=outf)
print('\n',file=outf)
print('1.Linear Kernel:',file=outf)
linear_param = {'kernel':['linear'],'gamma':['scale'],'C':[0.1,0.5,1,5,10,25,50,75,100,250,500]}
clf = GridSearchCV(SVC(),linear_param,scoring='accuracy',iid=False,cv=5)
# clf = svm.SVC(kernel = 'linear',gamma='scale')
clf.fit(fold1,fold1_label)
pre_fold2 = clf.predict(fold2);
pre_fold3 = clf.predict(fold3);
pre_fold4 = clf.predict(fold4);
acc_fold2 = accuracy_score(fold2_label,pre_fold2)
acc_fold3 = accuracy_score(fold3_label,pre_fold3)
acc_fold4 = accuracy_score(fold4_label,pre_fold4)
acc = [acc_fold2,acc_fold3,acc_fold4]
C = clf.best_params_['C']
print('The best parameters: C:%.2f'%C,file=outf)
print('accuracy:%f'%acc_fold2,file=outf)
print('The best parameters: C:%.2f'%C,file=outf)
print('accuracy:%f'%acc_fold3,file=outf)
print('The best parameters: C:%.2f'%C,file=outf)
print('accuracy:%f'%acc_fold4,file=outf)
print('Average 3-fold classification:',file=outf)
print('accuracy(along with standard deviation):%f'%np.mean(acc),file=outf)
print('(+%f)'%np.std(acc,ddof=1),file=outf)

print('\n',file=outf)
print('Gaussian Kernel:Now using rbf kernel.',file=outf)
rbf_param = {'kernel':['rbf'],'gamma': [1e-3, 1e-4],'C':[0.1,0.5,1,5,10,25,50,75,100,250,500]}
clf_rbf = GridSearchCV(SVC(),rbf_param,scoring='accuracy',iid = False,cv=5)
# clf_rbf = svm.SVC(kernel = 'rbf',gamma='scale')
clf_rbf.fit(fold1,fold1_label)
pre_fold2 = clf_rbf.predict(fold2);
pre_fold3 = clf_rbf.predict(fold3);
pre_fold4 = clf_rbf.predict(fold4);
acc_fold2 = accuracy_score(fold2_label,pre_fold2)
acc_fold3 = accuracy_score(fold3_label,pre_fold3)
acc_fold4 = accuracy_score(fold4_label,pre_fold4)
acc = [acc_fold2,acc_fold3,acc_fold4]
C = clf_rbf.best_params_['C']
gamma = clf_rbf.best_params_['gamma']
print('The best parameters: C:%.2f;gammma:%.2E'%(C,gamma),file=outf)
print('accuracy:%f'%acc_fold2,file=outf)
print('The best parameters: C:%.2f;gammma:%.2E'%(C,gamma),file=outf)
print('accuracy:%f'%acc_fold3,file=outf)
print('The best parameters: C:%.2f;gammma:%.2E'%(C,gamma),file=outf)
print('accuracy:%f'%acc_fold4,file=outf)
print('Average 3-fold classification:',file=outf)
print('accuracy(along with standard deviation):%f'%np.mean(acc),file=outf)
print('(+%f)'%np.std(acc,ddof=1),file=outf)


# In[ ]:




