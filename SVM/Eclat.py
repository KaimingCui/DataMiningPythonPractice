#!/usr/bin/env python
# coding: utf-8

# In[73]:
#Questions:

# a. Run the itemset mining algorithm with 20% support. How many frequent itemsets are there?
# b. Write top 10 itemsets (in terms of highest support value).
# c. How many frequent itemsets have 100 as part of itemsets?
# d. How many frequent itemsets have 200 as part of itemsets?
# e. Write top 10 association rules (in terms of highest confidence value) where the rule’s head is 100
# f. How many rules with head 100 are there for which the confidence value is more than 75%? List them. For this you need to write a small script that finds confidence of a rule from the support value of its body and the support value of its body plus head.
# g. Write top 10 association rules (in terms of highest confidence value) where the rule’s head is 200
# h. How many rules with head 200 are there for which the confidence value is more than 75%? List them.
# i. Use the rules (which has more than 75% confidence) as binary feature and construct a new dataset, in which each rule is a feature and any transaction that has the body of the rule will have a feature value of 1 and if it does not have the body of the rule will have a feature value 0. Use soft-margin SVM with linear kernel to report 3-fold classification accuracy (after using 1 fold for parameter tuning). Report both average and standard deviation.

# %load /Users/kaimingcui/Desktop/Semester3/573DataMining/assign4_iuKaimingCui/Eclat.py
import numpy as np
import fim
from fim import eclat
from fim import arules
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

def takeSecond(elem):
    return elem[1]

def takeFourth(elem):
    return elem[3]

f = open('house-votes-84.data','r')
outf = open('EclatResult.txt','w')
print('Part2:',file=outf)
lines = f.readlines()
dataset = list()
for line in lines:
    strpline = line.rstrip()
    arr = strpline.split(',')
    newline = [];
    for i in range(len(arr)):
        if arr[i] == 'y':
            newline.append(i)
    if arr[0] == 'republican':
        newline.append(100)
    else:
        newline.append(200)
    dataset.append(newline);

print("\npart a:",file=outf)
frequentset = eclat(dataset, supp=20, zmin=1)
print("with 20%% support, there are %d frequent itemsets."%len(frequentset),file=outf)

print("\npart b:",file=outf)
print("Top ten itemsets:",file=outf)
frequentset.sort(key=takeSecond,reverse = True)
i = 0;
for r in frequentset:
    if(i<10):
        print(r,file=outf);
        i += 1;
    else:
        break;

        
print("\npart c:",file=outf)
rec = 0
dec = 0
for r in frequentset:
    if 100 in r[0]:
        rec += 1
    elif 200 in r[0]:
        dec += 1

print("The number of frequent item that has 100:%d"%rec,file=outf)

print("\npart d:",file=outf)
print("The number of frequent item that has 200:%d"%dec,file=outf)

print("\npart e:",file=outf)
print("Top 10 rules with head 100:",file=outf)
print("output formation: rule_head|rule_tail|sup|confidence",file=outf)
rules = arules(dataset,supp=20,conf=0)
rule_100head = list()
for r in rules:
    if(r[0] == 100):
        rule_100head.append(r)
    elif(type(r[0])!=type(1)):
        if 100 in r[0]:
            rule_100head.append(r)
rule_100head.sort(key=takeFourth,reverse=True)
i = 0
for r in rule_100head:
    if(i<10):
        print(r,file=outf)
        i+=1
    else:
        break;

print("\npart f:",file=outf)
templist = list()
for r in rule_100head:
    if(r[3] >= 75):
        templist.append(r)
print("The number of rules with head 100 and confidence value more than 75%% (greater or equal to 75%%):%d"%len(templist),file=outf)

print("\npart g:",file=outf)
print("Top 10 rules with head 200:",file=outf)
print("output formation: rule_head|rule_tail|sup|confidence",file=outf)
rule_200head = list()
for r in rules:
    if(r[0] == 200):
        rule_200head.append(r)
    elif(type(r[0])!=type(1)):
        if 200 in r[0]:
            rule_200head.append(r)
rule_200head.sort(key=takeFourth,reverse=True)
i = 0
for r in rule_200head:
    if(i<10):
        print(r,file=outf)
        i+=1
    else:
        break;

print("\npart h:",file=outf)
templist = list()
for r in rule_200head:
    if(r[3] >= 75):
        templist.append(r)
print("The number of rules with head 100 and confidence value more than 75%% (greater or equal to 75%%):%d"%len(templist),file=outf)

print("\npart i:",file=outf)
print("Use linear kernel SVM classfication:",file=outf)
rules_75 = arules(dataset,supp=20,conf=75)
new_dataset = list()
for d in dataset:
    newdata_line = list()
    for r in rules_75:
        if(set(r[1]).issubset(d)):
            newdata_line.append(1)
        else:
            newdata_line.append(0)
    new_dataset.append(newdata_line)
# print(len(new_dataset))

fold1 = list()
fold2 = list()
fold3 = list()
fold4 = list()

fold1_label = list();
fold2_label = list();
fold3_label = list();
fold4_label = list();

fold_re_c = 0;
fold_de_c = 0;
for i in range(len(new_dataset)):
    if(dataset[i][-1] == 100):
        fold_re_c += 1;
        if(fold_re_c <= 42):
            fold1.append(new_dataset[i]);
            fold1_label.append(dataset[i][-1])
        elif(fold_re_c <= 84):
            fold2.append(new_dataset[i]);
            fold2_label.append(dataset[i][-1])
        elif(fold_re_c <= 126):
            fold3.append(new_dataset[i]);
            fold3_label.append(dataset[i][-1])
        else:
            fold4.append(new_dataset[i]);
            fold4_label.append(dataset[i][-1])
    else:
        fold_de_c += 1;
        if(fold_de_c <= 66):
            fold1.append(new_dataset[i]);
            fold1_label.append(dataset[i][-1])
        elif(fold_de_c <= 133):
            fold2.append(new_dataset[i]);
            fold2_label.append(dataset[i][-1])
        elif(fold_de_c <= 200):
            fold3.append(new_dataset[i]);
            fold3_label.append(dataset[i][-1])
        else:
            fold4.append(new_dataset[i]);
            fold4_label.append(dataset[i][-1])


linear_param = {'kernel':['linear'],'gamma':['scale'],'C':[0.1,0.5,1,5,10,25,50,75,100,250,500]}
clf = GridSearchCV(SVC(),linear_param,scoring='accuracy',iid=False,cv=5)
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

