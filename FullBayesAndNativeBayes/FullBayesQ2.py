import numpy.matlib
import numpy.linalg
import numpy as np
import sys
import scipy.stats 
from scipy.stats import multivariate_normal

#full bayes
#compute likelihood
def likelihood(data,means,covars):
	pdf = multivariate_normal.pdf(data,means,covars)
	return pdf

#classifier
def full_bayes_classifier(data,means,covars,P):
	classes = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
	temp = list()
	for i in range(3):
		pdf = likelihood(data,means[i],covars[i])
		arg = pdf*P[i]
		temp.append(arg)
	maxindex = temp.index(max(temp))
	return classes[maxindex]

# read the data from model file and convert data into correct format
modelfile = sys.argv[1]
textfile = sys.argv[2]
#data = np.loadtxt(textfile,delimiter = ',',usecols = np.arange(0,4))
#true_label = np.loadtxt(textfile,dtype = bytes,delimiter = ',',usecols = (4,)).astype(str)
test_data = np.loadtxt(textfile,delimiter = ',',usecols = np.arange(0,4))
true_label = np.loadtxt(textfile,dtype = bytes,delimiter = ',',usecols = (4,)).astype(str)
class_label = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
model_data = list()
#f = open(modelfile,'r')
f = open(modelfile,'r')
for line in f.readlines():
	model_data.append(line.strip('\n'))
# print(model_data)
prior = list()
i = 0
for i in  range(0,len(model_data),3):
	prior.append(float(model_data[i]))
# print(prior)

means = list()
for i in  range(1,len(model_data),3):
	temp = model_data[i].strip('[]').split(',')
	mean = list()
	for item in temp:
		mean.append(float(item))
	means.append(np.array(mean))



covs = list()
one_cov = list()
for i in  range(2,len(model_data),3):
	temp = model_data[i]
	temp = temp.strip('[]').split(',')
	temp_cov = list()
	counter = 0
	for item in temp:
		temp_cov.append(float(item.strip().strip('[]')))
		counter+=1
		if(counter%4 == 0):
			one_cov.append(temp_cov)
			temp_cov = []
	covs.append(np.array(one_cov))
	one_cov = []


rows = test_data.shape[0]
temp_label = list()
print("-----------------------------------------")
print("Each point and its perdicted label by full bayes classifier:")
for i in range(rows):
	result = full_bayes_classifier(test_data[i],means,covs,prior)
	temp_label.append(result)
	print(test_data[i],end=",")
	print(result)
result_label = np.array(temp_label)
# print(result_label)

#compute the confusion matrix
confusion = np.zeros((3,3))
for i in range(rows):
	if(result_label[i] == class_label[0]):
			if(true_label[i] == class_label[0]):
				confusion[0][0] += 1
			elif(true_label[i] == class_label[1]):
				confusion[0][1] += 1
			elif(true_label[i] == class_label[2]):
				confusion[0][2] += 1
	elif(result_label[i] == class_label[1]):
			if(true_label[i] == class_label[0]):
				confusion[1][0] += 1
			elif(true_label[i] == class_label[1]):
				confusion[1][1] += 1
			elif(true_label[i] == class_label[2]):
				confusion[1][2] += 1
	elif(result_label[i] == class_label[2]):
			if(true_label[i] == class_label[0]):
				confusion[2][0] += 1
			elif(true_label[i] == class_label[1]):
				confusion[2][1] += 1
			elif(true_label[i] == class_label[2]):
				confusion[2][2] += 1
print("-----------------------------------------")
print("The confusion matrix:")
print(confusion)
f.close()