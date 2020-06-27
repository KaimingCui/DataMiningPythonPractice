import numpy.matlib
import numpy.linalg
import numpy as np
import sys
import scipy.stats 
from scipy.stats import multivariate_normal

filename = sys.argv[1]
#data = np.loadtxt(filename,delimiter = ',',usecols = np.arange(0,4))
#data_str = np.loadtxt(filename,dtype = bytes,delimiter = ',',usecols = (4,)).astype(str)
data = np.loadtxt(filename,delimiter = ',',usecols = np.arange(0,4))
data_str =  np.loadtxt(filename,dtype = bytes,delimiter = ',',usecols = (4,)).astype(str)
class_label = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
f = open("model.txt",'w+')
rows = data.shape[0]
c1 = 0
c2 = 0
c3 = 0
for i in range(rows):
	if(data_str[i] == class_label[0]):
		c1+=1
	elif(data_str[i] == class_label[1]):
		c2+=1
	#Iris-virginica
	else:
		c3+=1

class1 = np.zeros((c1,4))
counter1 = 0
class2 = np.zeros((c2,4))
counter2 = 0
class3 = np.zeros((c3,4))
counter3 = 0

i = 0
for i in range(rows):
	if(data_str[i] == class_label[0]):
		class1[counter1] = data[i]
		counter1 += 1
	elif(data_str[i] == class_label[1]):
		class2[counter2] = data[i]
		counter2 += 1
	else:
		class3[counter3] = data[i]
		counter3 += 1

classes = list()
classes.append(class1)
classes.append(class2)
classes.append(class3)

for i in range(3):
	prior = round(classes[i].shape[0]/rows,2)
	mean = np.around(np.mean(classes[i],axis=0),decimals = 2)
	cov_matrix = np.around(np.cov(classes[i],rowvar = 0, bias = 1),decimals = 2)
	variances = cov_matrix.diagonal()
	cov = np.diag(variances)
	print(prior,file = f)
	print(mean.tolist(),file = f)
	print(cov.tolist(),file= f)

f.close()