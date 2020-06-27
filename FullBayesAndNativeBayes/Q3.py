import numpy.matlib
import numpy.linalg
import numpy as np
import sys
import scipy.stats 
from scipy.stats import multivariate_normal

#get likelihood
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

#navie
def likelihood_perDimension(xj,meanij,covij):
	pdfij = multivariate_normal.pdf(xj,meanij,covij)
	return pdfij


def likelihood_navie(data,means,covars):
	variances = covars.diagonal()
	pdf = 1
	for i in range(4):
		temp = likelihood_perDimension(data[i],means[i],variances[i])
		pdf = pdf*temp
	return pdf

#classifier
def navie_bayes_classifier(data,means,covars,P):
	classes = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
	temp = list()
	for i in range(3):
		pdf = likelihood_navie(data,means[i],covars[i])
		arg = pdf*P[i]
		temp.append(arg)
	maxindex = temp.index(max(temp))
	return classes[maxindex]


def compute_accuracy(confusion,N):
	accuracy = list()
	correct = confusion.diagonal()
	i = 0
	for i in range(3):
		accuracy.append(correct[i]/N[i])
	# for item in correct:
	# 	i  = i + item
	# accuracy = i/N
	return accuracy

def compute_Pr(confusion):
	pr = list()
	i = 0
	for i in range(confusion.shape[0]):
		j = 0
		temp = 0
		for j in  range(confusion.shape[1]):
			temp = temp + confusion[i][j]
		per_pr = confusion[i][i]/temp
		pr.append(per_pr)
	return pr

def compute_Re(confusion):
	re = list()
	i = 0
	for i in range(confusion.shape[1]):
		j = 0
		temp = 0
		for j in range(confusion.shape[0]):
			temp = temp + confusion[j][i]
		per_re = confusion[i][i]/temp
		re.append(per_re)
	return re

def compute_f_score(pr,re):
	f_score = list()
	i = 0
	for i in range(len(pr)):
		temp = (2*pr[i]*re[i])/(pr[i]+re[i])
		f_score.append(temp)
	return f_score

#get data and folds
f = open("Q3Answers.txt",'w')
data = np.loadtxt("iris.txt.shuffled",delimiter = ',',usecols = np.arange(0,4))
labels = np.loadtxt("iris.txt.shuffled",dtype = bytes,delimiter = ',',usecols = (4,)).astype(str)
class_label = {0:'Iris-setosa',1:'Iris-versicolor',2:'Iris-virginica'}
print("indicate class:",file=f)
print("class 1: Iris-setosa",file=f)
print("class 1: Iris-versicolor",file=f)
print("class 1: Iris-virginica",file=f)
fold1 = np.zeros((50,4))
fold2 = np.zeros((50,4))
fold3 = np.zeros((50,4))

fold1 = data[0:50,:]
fold2 = data[50:100,:]
fold3 = data[100:150,:]
# folds = list()
# folds.append(fold1)
# folds.append(fold2)
# folds.append(fold3)

fold1_true_label = labels[0:50]
fold2_true_label = labels[50:100]
fold3_true_label = labels[100:150]
# folds_label = list()
# folds_label.append(fold1_true_label)
# folds_label.append(fold2_true_label)
# folds_label.append(fold3_true_label)

prior = list()
means = list()
covs = list()
print("-----------------------------------------",file = f)
print("Answer of Q3 for full bayes classifier:",file = f)
avg_over3 = list()
# print("-------------------------------------------------------------------")
# print('\n')
#fold 1 and 2 as tranining data
training_data = np.append(fold1,fold2,axis = 0)
tranining_data_label = np.append(fold1_true_label,fold2_true_label,axis = 0)
test_data = fold3
#Q1
rows = training_data.shape[0]
c1 = 0
c2 = 0
c3 = 0
for i in range(rows):
	if(tranining_data_label[i] == class_label[0]):
		c1+=1
	elif(tranining_data_label[i] == class_label[1]):
		c2+=1
	#class_label[2]
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
	if(tranining_data_label[i] == class_label[0]):
		class1[counter1] = training_data[i]
		counter1 += 1
	elif(tranining_data_label[i] == class_label[1]):
		class2[counter2] = training_data[i]
		counter2 += 1
	else:
		class3[counter3] = training_data[i]
		counter3 += 1
classes = list()
classes.append(class1)
classes.append(class2)
classes.append(class3)

for i in range(3):
	a_prior = round(classes[i].shape[0]/rows,2)
	a_mean = np.around(np.mean(classes[i],axis=0),decimals = 2)
	a_cov = np.around(np.cov(classes[i],rowvar = 0, bias = 1),decimals = 2)
	prior.append(a_prior)
	means.append(a_mean)
	covs.append(a_cov)
# print(prior)
# print(means)
# print(covs)
#Q2
test_data_rows = test_data.shape[0]
temp_label = list()
for i in range(test_data_rows):
	result = full_bayes_classifier(test_data[i],means,covs,prior)
	temp_label.append(result)
result_label = np.array(temp_label)

c1 = 0
c2 = 0
c3 = 0
for i in range(test_data_rows):
	if(fold3_true_label[i] == class_label[0]):
		c1+=1
	elif(fold3_true_label[i] == class_label[1]):
		c2+=1
	else:
		c3+=1
class_static = list()
class_static.append(c1)
class_static.append(c2)
class_static.append(c3)


confusion = np.zeros((3,3))
for i in range(test_data_rows):
	if(result_label[i] == class_label[0]):
			if(fold3_true_label[i] == class_label[0]):
				confusion[0][0] += 1
			elif(fold3_true_label[i] == class_label[1]):
				confusion[0][1] += 1
			elif(fold3_true_label[i] == class_label[2]):
				confusion[0][2] += 1
	elif(result_label[i] == class_label[1]):
			if(fold3_true_label[i] == class_label[0]):
				confusion[1][0] += 1
			elif(fold3_true_label[i] == class_label[1]):
				confusion[1][1] += 1
			elif(fold3_true_label[i] == class_label[2]):
				confusion[1][2] += 1
	elif(result_label[i] == class_label[2]):
			if(fold3_true_label[i] == class_label[0]):
				confusion[2][0] += 1
			elif(fold3_true_label[i] == class_label[1]):
				confusion[2][1] += 1
			elif(fold3_true_label[i] == class_label[2]):
				confusion[2][2] += 1
print("-----------------------------------------",file = f)
print("Fold 3: fold 3 is tested, fold 1 and fold 2 are training data.",file = f);
print("The confusion matrix when fold 3 is tested:",file = f)
print(confusion,file = f)
print("Classification Report:",file = f)
print("Class: accuracy|precision|recall|fscore",file = f)
fold3_acc_full = compute_accuracy(confusion,class_static)
fold3_pr_full = compute_Pr(confusion)
fold3_re_full = compute_Re(confusion)
fold3_fscore_full = compute_f_score(fold3_pr_full,fold3_re_full)
print("1: [%.2f %.2f %.2f %.2f]"%(fold3_acc_full[0],fold3_pr_full[0],fold3_re_full[0],fold3_fscore_full[0]),file = f)
print("2: [%.2f %.2f %.2f %.2f]"%(fold3_acc_full[1],fold3_pr_full[1],fold3_re_full[1],fold3_fscore_full[1]),file = f)
print("3: [%.2f %.2f %.2f %.2f]"%(fold3_acc_full[2],fold3_pr_full[2],fold3_re_full[2],fold3_fscore_full[2]),file = f)
avgacc3 = (fold3_acc_full[0]+fold3_acc_full[1]+fold3_acc_full[2])/3
avgpr3 = (fold3_pr_full[0]+fold3_pr_full[1]+fold3_pr_full[2])/3
avgre3 = (fold3_re_full[0]+fold3_re_full[1]+fold3_re_full[2])/3
avgfscore3 = (fold3_fscore_full[0]+fold3_fscore_full[1]+fold3_fscore_full[2])/3
print("avg: [%.2f %.2f %.2f %.2f]"%(avgacc3,avgpr3,avgre3,avgfscore3),file = f)
print('\n',file = f)



#fold 1 and 3 as tranining data
training_data = np.append(fold1,fold3,axis = 0)
tranining_data_label = np.append(fold1_true_label,fold3_true_label,axis = 0)
test_data = fold2
prior = list()
means = list()
covs = list()
#Q1
rows = training_data.shape[0]
c1 = 0
c2 = 0
c3 = 0
for i in range(rows):
	if(tranining_data_label[i] == class_label[0]):
		c1+=1
	elif(tranining_data_label[i] == class_label[1]):
		c2+=1
	#class_label[2]
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
	if(tranining_data_label[i] == class_label[0]):
		class1[counter1] = training_data[i]
		counter1 += 1
	elif(tranining_data_label[i] == class_label[1]):
		class2[counter2] = training_data[i]
		counter2 += 1
	else:
		class3[counter3] = training_data[i]
		counter3 += 1
classes = list()
classes.append(class1)
classes.append(class2)
classes.append(class3)

for i in range(3):
	a_prior = round(classes[i].shape[0]/rows,2)
	a_mean = np.around(np.mean(classes[i],axis=0),decimals = 2)
	a_cov = np.around(np.cov(classes[i],rowvar = 0, bias = 1),decimals = 2)
	prior.append(a_prior)
	means.append(a_mean)
	covs.append(a_cov)
# print(prior)
# print(means)
# print(covs)
#Q2
test_data_rows = test_data.shape[0]
temp_label = list()
for i in range(test_data_rows):
	result = full_bayes_classifier(test_data[i],means,covs,prior)
	temp_label.append(result)
result_label = np.array(temp_label)

c1 = 0
c2 = 0
c3 = 0
for i in range(test_data_rows):
	if(fold2_true_label[i] == class_label[0]):
		c1+=1
	elif(fold2_true_label[i] == class_label[1]):
		c2+=1
	else:
		c3+=1
class_static = list()
class_static.append(c1)
class_static.append(c2)
class_static.append(c3)


confusion = np.zeros((3,3))
for i in range(test_data_rows):
	if(result_label[i] == class_label[0]):
			if(fold2_true_label[i] == class_label[0]):
				confusion[0][0] += 1
			elif(fold2_true_label[i] == class_label[1]):
				confusion[0][1] += 1
			elif(fold2_true_label[i] == class_label[2]):
				confusion[0][2] += 1
	elif(result_label[i] == class_label[1]):
			if(fold2_true_label[i] == class_label[0]):
				confusion[1][0] += 1
			elif(fold2_true_label[i] == class_label[1]):
				confusion[1][1] += 1
			elif(fold2_true_label[i] == class_label[2]):
				confusion[1][2] += 1
	elif(result_label[i] == class_label[2]):
			if(fold2_true_label[i] == class_label[0]):
				confusion[2][0] += 1
			elif(fold2_true_label[i] == class_label[1]):
				confusion[2][1] += 1
			elif(fold2_true_label[i] == class_label[2]):
				confusion[2][2] += 1
print("-----------------------------------------",file = f)
print("Fold 2: fold 2 is tested, fold 1 and fold 3 are training data.",file = f);
print("The confusion matrix when fold 2 is tested:",file = f)
print(confusion,file = f)
print("Classification Report:",file = f)
print("Class: accuracy|precision|recall|fscore",file = f)
fold2_acc_full = compute_accuracy(confusion,class_static)
fold2_pr_full = compute_Pr(confusion)
fold2_re_full = compute_Re(confusion)
fold2_fscore_full = compute_f_score(fold2_pr_full,fold2_re_full)
print("1: [%.2f %.2f %.2f %.2f]"%(fold2_acc_full[0],fold2_pr_full[0],fold2_re_full[0],fold2_fscore_full[0]),file = f)
print("2: [%.2f %.2f %.2f %.2f]"%(fold2_acc_full[1],fold2_pr_full[1],fold2_re_full[1],fold2_fscore_full[1]),file = f)
print("3: [%.2f %.2f %.2f %.2f]"%(fold2_acc_full[2],fold2_pr_full[2],fold2_re_full[2],fold2_fscore_full[2]),file = f)
avgacc2 = (fold2_acc_full[0]+fold2_acc_full[1]+fold2_acc_full[2])/3
avgpr2 = (fold2_pr_full[0]+fold2_pr_full[1]+fold2_pr_full[2])/3
avgre2 = (fold2_re_full[0]+fold2_re_full[1]+fold2_re_full[2])/3
avgfscore2 = (fold2_fscore_full[0]+fold2_fscore_full[1]+fold2_fscore_full[2])/3
print("avg: [%.2f %.2f %.2f %.2f]"%(avgacc2,avgpr2,avgre2,avgfscore2),file = f)
print('\n',file = f)


#fold 2 and 3 as tranining data
prior = list()
means = list()
covs = list()
training_data = np.append(fold2,fold3,axis = 0)
tranining_data_label = np.append(fold2_true_label,fold3_true_label,axis = 0)
test_data = fold1
#Q1
rows = training_data.shape[0]
c1 = 0
c2 = 0
c3 = 0
for i in range(rows):
	if(tranining_data_label[i] == class_label[0]):
		c1+=1
	elif(tranining_data_label[i] == class_label[1]):
		c2+=1
	#class_label[2]
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
	if(tranining_data_label[i] == class_label[0]):
		class1[counter1] = training_data[i]
		counter1 += 1
	elif(tranining_data_label[i] == class_label[1]):
		class2[counter2] = training_data[i]
		counter2 += 1
	else:
		class3[counter3] = training_data[i]
		counter3 += 1
classes = list()
classes.append(class1)
classes.append(class2)
classes.append(class3)
# print(classes)

for i in range(3):
	a_prior = round(classes[i].shape[0]/rows,2)
	a_mean = np.around(np.mean(classes[i],axis=0),decimals = 2)
	a_cov = np.around(np.cov(classes[i],rowvar = 0, bias = 1),decimals = 2)
	prior.append(a_prior)
	means.append(a_mean)
	covs.append(a_cov)
# print(prior)
# print(means)
# print(covs)
#Q2
test_data_rows = test_data.shape[0]
temp_label = list()
for i in range(test_data_rows):
	result = full_bayes_classifier(test_data[i],means,covs,prior)
	temp_label.append(result)
result_label = np.array(temp_label)


c1 = 0
c2 = 0
c3 = 0
for i in range(test_data_rows):
	if(fold1_true_label[i] == class_label[0]):
		c1+=1
	elif(fold1_true_label[i] == class_label[1]):
		c2+=1
	else:
		c3+=1
class_static = list()
class_static.append(c1)
class_static.append(c2)
class_static.append(c3)



confusion = np.zeros((3,3))
for i in range(test_data_rows):
	if(result_label[i] == class_label[0]):
			if(fold1_true_label[i] == class_label[0]):
				confusion[0][0] += 1
			elif(fold1_true_label[i] == class_label[1]):
				confusion[0][1] += 1
			elif(fold1_true_label[i] == class_label[2]):
				confusion[0][2] += 1
	elif(result_label[i] == class_label[1]):
			if(fold1_true_label[i] == class_label[0]):
				confusion[1][0] += 1
			elif(fold1_true_label[i] == class_label[1]):
				confusion[1][1] += 1
			elif(fold1_true_label[i] == class_label[2]):
				confusion[1][2] += 1
	elif(result_label[i] == class_label[2]):
			if(fold1_true_label[i] == class_label[0]):
				confusion[2][0] += 1
			elif(fold1_true_label[i] == class_label[1]):
				confusion[2][1] += 1
			elif(fold1_true_label[i] == class_label[2]):
				confusion[2][2] += 1
print("-----------------------------------------",file = f)
print("Fold 1: fold 1 is tested, fold 2 and fold 3 are training data.",file = f);
print("The confusion matrix when fold 1 is tested:",file = f)
print(confusion,file = f)
print("Classification Report:",file = f)
print("Class: accuracy|precision|recall|fscore",file = f)
fold1_acc_full = compute_accuracy(confusion,class_static)
fold1_pr_full = compute_Pr(confusion)
fold1_re_full = compute_Re(confusion)
fold1_fscore_full = compute_f_score(fold1_pr_full,fold1_re_full)
print("1: [%.2f %.2f %.2f %.2f]"%(fold1_acc_full[0],fold1_pr_full[0],fold1_re_full[0],fold1_fscore_full[0]),file = f)
print("2: [%.2f %.2f %.2f %.2f]"%(fold1_acc_full[1],fold1_pr_full[1],fold1_re_full[1],fold1_fscore_full[1]),file = f)
print("3: [%.2f %.2f %.2f %.2f]"%(fold1_acc_full[2],fold1_pr_full[2],fold1_re_full[2],fold1_fscore_full[2]),file = f)
avgacc1 = (fold1_acc_full[0]+fold1_acc_full[1]+fold1_acc_full[2])/3
avgpr1 = (fold1_pr_full[0]+fold1_pr_full[1]+fold1_pr_full[2])/3
avgre1 = (fold1_re_full[0]+fold1_re_full[1]+fold1_re_full[2])/3
avgfscore1= (fold1_fscore_full[0]+fold1_fscore_full[1]+fold1_fscore_full[2])/3
print("avg: [%.2f %.2f %.2f %.2f]"%(avgacc1,avgpr1,avgre1,avgfscore1),file = f)
print('\n',file = f)



i = 0
# # avr_acc = (fold1_acc_full+fold2_acc_full+fold3_acc_full)/3
# avr_pr = list()
# avr_re = list()
# avr_fscore = list()
# for i in range(confusion.shape[0]):
# 	pr = (fold1_pr_full[i]+fold2_pr_full[i]+fold3_pr_full[i])/3
# 	re = (fold1_re_full[i]+fold2_re_full[i]+fold3_re_full[i])/3
# 	fscore = (fold1_fscore_full[i]+fold2_fscore_full[i]+fold3_fscore_full[i])/3
# 	avr_pr.append(pr)
# 	avr_re.append(re)
# 	avr_fscore.append(fscore)
print("-----------------------------------------",file = f)
print("Averaged value of evaluation over 3 folds:",file=f)
print("Class:accuracy|precision|recall|fscore",file = f)
avgacc_over = (avgacc1+avgacc2+avgacc3)/3
avgpr_over = (avgpr1+avgpr2+avgpr3)/3
avgre_over = (avgre1+avgre2+avgre3)/3
avgfs_over = (avgfscore1+avgfscore2+avgfscore3)/3
print("Avg: [%.2f %.2f %.2f %.2f]"%(avgacc_over,avgpr_over,avgre_over,avgfs_over),file = f)
print("-----------------------------------------",file = f)

print("Averaged value of evaluation in each class:",file=f)
print("Class:accuracy|precision|recall|fscore",file = f)
Avgacc_c1 = list()
for i in range(3):
	acc = (fold1_acc_full[i]+fold2_acc_full[i]+fold3_acc_full[i])/3
	pr = (fold1_pr_full[i]+fold2_pr_full[i]+fold3_pr_full[i])/3
	re = (fold1_re_full[i]+fold2_re_full[i]+fold3_re_full[i])/3
	fs = (fold1_fscore_full[i]+fold2_fscore_full[i]+fold3_fscore_full[i])/3
	print("Avg_c%d: [%.2f %.2f %.2f %.2f]"%((i+1),acc,pr,re,fs),file = f)
print("-----------------------------------------",file = f)
# print("The accuracy averaged over 3 folds is:",file = f)
# print(round(avr_acc,2),file = f)
# print("The precision for each class averaged over 3 folds is:",file = f)
# for i in range(confusion.shape[0]):
# 	print("%s    %.2f"%(class_label[i],round(avr_pr[i],2)),file = f)
# print("The recall for each class averaged over 3 folds is:",file = f)
# for i in range(confusion.shape[0]):
# 	print("%s    %.2f"%(class_label[i],round(avr_re[i],2)),file = f)
# print("The f-score for each class averaged over 3 folds is:",file = f)
# for i in range(confusion.shape[0]):
# 	print("%s    %.2f"%(class_label[i],round(avr_fscore[i],2)),file = f)


print('\n',file = f)
print("Answer of Q3 for navie bayes classifier:",file = f)
# print("-------------------------------------------------------------------")
# print('\n')

#fold 1 and 2 as tranining data
prior = list()
means = list()
covs = list()
training_data = np.append(fold1,fold2,axis = 0)
tranining_data_label = np.append(fold1_true_label,fold2_true_label,axis = 0)
test_data = fold3
#Q1
rows = training_data.shape[0]
test_data_rows = test_data.shape[0]
c1 = 0
c2 = 0
c3 = 0
for i in range(rows):
	if(tranining_data_label[i] == class_label[0]):
		c1+=1
	elif(tranining_data_label[i] == class_label[1]):
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
	if(tranining_data_label[i] == class_label[0]):
		class1[counter1] = training_data[i]
		counter1 += 1
	elif(tranining_data_label[i] == class_label[1]):
		class2[counter2] = training_data[i]
		counter2 += 1
	else:
		class3[counter3] = training_data[i]
		counter3 += 1

classes = list()
classes.append(class1)
classes.append(class2)
classes.append(class3)

for i in range(3):
	a_prior = round(classes[i].shape[0]/rows,2)
	mean = np.around(np.mean(classes[i],axis=0),decimals = 2)
	cov_matrix = np.around(np.cov(classes[i],rowvar = 0, bias = 1),decimals = 2)
	variances = cov_matrix.diagonal()
	cov = np.diag(variances)
	prior.append(a_prior)
	means.append(mean)
	covs.append(cov)
	# print(prior,file = f)
	# print(mean.tolist(),file = f)
	# print(cov.tolist(),file= f)

temp_label = list()
for i in range(test_data_rows):
	result = navie_bayes_classifier(test_data[i],means,covs,prior)
	temp_label.append(result)
	# print(test_data[i],end=",")
	# print(result)
result_label = np.array(temp_label)

c1 = 0
c2 = 0
c3 = 0
for i in range(test_data_rows):
	if(fold3_true_label[i] == class_label[0]):
		c1+=1
	elif(fold3_true_label[i] == class_label[1]):
		c2+=1
	else:
		c3+=1
class_static = list()
class_static.append(c1)
class_static.append(c2)
class_static.append(c3)

#compute the confusion matrix
confusion = np.zeros((3,3))
for i in range(test_data_rows):
	if(result_label[i] == class_label[0]):
			if(fold3_true_label[i] == class_label[0]):
				confusion[0][0] += 1
			elif(fold3_true_label[i] == class_label[1]):
				confusion[0][1] += 1
			elif(fold3_true_label[i] == class_label[2]):
				confusion[0][2] += 1
	elif(result_label[i] == class_label[1]):
			if(fold3_true_label[i] == class_label[0]):
				confusion[1][0] += 1
			elif(fold3_true_label[i] == class_label[1]):
				confusion[1][1] += 1
			elif(fold3_true_label[i] == class_label[2]):
				confusion[1][2] += 1
	elif(result_label[i] == class_label[2]):
			if(fold3_true_label[i] == class_label[0]):
				confusion[2][0] += 1
			elif(fold3_true_label[i] == class_label[1]):
				confusion[2][1] += 1
			elif(fold3_true_label[i] == class_label[2]):
				confusion[2][2] += 1
print("-----------------------------------------",file = f)
print("Fold 3: fold 3 is tested, fold 1 and fold 2 are training data.",file = f);
print("The confusion matrix when fold 3 is tested:",file = f)
print(confusion,file = f)
print("Classification Report:",file = f)
print("Class: accuracy|precision|recall|fscore",file = f)
fold3_acc_navie = compute_accuracy(confusion,class_static)
fold3_pr_navie = compute_Pr(confusion)
fold3_re_navie = compute_Re(confusion)
fold3_fscore_navie = compute_f_score(fold3_pr_navie,fold3_re_navie)
print("1: [%.2f %.2f %.2f %.2f]"%(fold3_acc_navie[0],fold3_pr_navie[0],fold3_re_navie[0],fold3_fscore_navie[0]),file = f)
print("2: [%.2f %.2f %.2f %.2f]"%(fold3_acc_navie[1],fold3_pr_navie[1],fold3_re_navie[1],fold3_fscore_navie[1]),file = f)
print("3: [%.2f %.2f %.2f %.2f]"%(fold3_acc_navie[2],fold3_pr_navie[2],fold3_re_navie[2],fold3_fscore_navie[2]),file = f)
avgacc3 = (fold3_acc_navie[0]+fold3_acc_navie[1]+fold3_acc_navie[2])/3
avgpr3 = (fold3_pr_navie[0]+fold3_pr_navie[1]+fold3_pr_navie[2])/3
avgre3 = (fold3_re_navie[0]+fold3_re_navie[1]+fold3_re_navie[2])/3
avgfscore3 = (fold3_fscore_navie[0]+fold3_fscore_navie[1]+fold3_fscore_navie[2])/3
print("avg: [%.2f %.2f %.2f %.2f]"%(avgacc3,avgpr3,avgre3,avgfscore3),file = f)
print('\n',file = f)



#fold 1 and 3 as tranining data
prior = list()
means = list()
covs = list()
training_data = np.append(fold1,fold3,axis = 0)
tranining_data_label = np.append(fold1_true_label,fold3_true_label,axis = 0)
test_data = fold2
#Q1
rows = training_data.shape[0]
test_data_rows = test_data.shape[0]
c1 = 0
c2 = 0
c3 = 0
for i in range(rows):
	if(tranining_data_label[i] == class_label[0]):
		c1+=1
	elif(tranining_data_label[i] == class_label[1]):
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
	if(tranining_data_label[i] == class_label[0]):
		class1[counter1] = training_data[i]
		counter1 += 1
	elif(tranining_data_label[i] == class_label[1]):
		class2[counter2] = training_data[i]
		counter2 += 1
	else:
		class3[counter3] = training_data[i]
		counter3 += 1

classes = list()
classes.append(class1)
classes.append(class2)
classes.append(class3)

for i in range(3):
	a_prior = round(classes[i].shape[0]/rows,2)
	mean = np.around(np.mean(classes[i],axis=0),decimals = 2)
	cov_matrix = np.around(np.cov(classes[i],rowvar = 0, bias = 1),decimals = 2)
	variances = cov_matrix.diagonal()
	cov = np.diag(variances)
	prior.append(a_prior)
	means.append(mean)
	covs.append(cov)
	# print(prior,file = f)
	# print(mean.tolist(),file = f)
	# print(cov.tolist(),file= f)

temp_label = list()
for i in range(test_data_rows):
	result = navie_bayes_classifier(test_data[i],means,covs,prior)
	temp_label.append(result)
	# print(test_data[i],end=",")
	# print(result)
result_label = np.array(temp_label)

c1 = 0
c2 = 0
c3 = 0
for i in range(test_data_rows):
	if(fold2_true_label[i] == class_label[0]):
		c1+=1
	elif(fold2_true_label[i] == class_label[1]):
		c2+=1
	else:
		c3+=1
class_static = list()
class_static.append(c1)
class_static.append(c2)
class_static.append(c3)

#compute the confusion matrix
confusion = np.zeros((3,3))
for i in range(test_data_rows):
	if(result_label[i] == class_label[0]):
			if(fold2_true_label[i] == class_label[0]):
				confusion[0][0] += 1
			elif(fold2_true_label[i] == class_label[1]):
				confusion[0][1] += 1
			elif(fold2_true_label[i] == class_label[2]):
				confusion[0][2] += 1
	elif(result_label[i] == class_label[1]):
			if(fold2_true_label[i] == class_label[0]):
				confusion[1][0] += 1
			elif(fold2_true_label[i] == class_label[1]):
				confusion[1][1] += 1
			elif(fold2_true_label[i] == class_label[2]):
				confusion[1][2] += 1
	elif(result_label[i] == class_label[2]):
			if(fold2_true_label[i] == class_label[0]):
				confusion[2][0] += 1
			elif(fold2_true_label[i] == class_label[1]):
				confusion[2][1] += 1
			elif(fold2_true_label[i] == class_label[2]):
				confusion[2][2] += 1
print("-----------------------------------------",file = f)
print("Fold 2: fold 2 is tested, fold 1 and fold 3 are training data.",file = f);
print("The confusion matrix when fold 2 is tested:",file = f)
print(confusion,file = f)
print("Classification Report:",file = f)
print("Class: accuracy|precision|recall|fscore",file = f)
fold2_acc_navie = compute_accuracy(confusion,class_static)
fold2_pr_navie = compute_Pr(confusion)
fold2_re_navie = compute_Re(confusion)
fold2_fscore_navie = compute_f_score(fold2_pr_navie,fold2_re_navie)
print("1: [%.2f %.2f %.2f %.2f]"%(fold2_acc_navie[0],fold2_pr_navie[0],fold2_re_navie[0],fold2_fscore_navie[0]),file = f)
print("2: [%.2f %.2f %.2f %.2f]"%(fold2_acc_navie[1],fold2_pr_navie[1],fold2_re_navie[1],fold2_fscore_navie[1]),file = f)
print("3: [%.2f %.2f %.2f %.2f]"%(fold2_acc_navie[2],fold2_pr_navie[2],fold2_re_navie[2],fold2_fscore_navie[2]),file = f)
avgacc2 = (fold2_acc_navie[0]+fold2_acc_navie[1]+fold2_acc_navie[2])/3
avgpr2 = (fold2_pr_navie[0]+fold2_pr_navie[1]+fold2_pr_navie[2])/3
avgre2 = (fold2_re_navie[0]+fold2_re_navie[1]+fold2_re_navie[2])/3
avgfscore2 = (fold2_fscore_navie[0]+fold2_fscore_navie[1]+fold2_fscore_navie[2])/3
print("avg: [%.2f %.2f %.2f %.2f]"%(avgacc2,avgpr2,avgre2,avgfscore2),file = f)
print('\n',file = f)




#fold 2 and 3 as tranining data
prior = list()
means = list()
covs = list()
training_data = np.append(fold2,fold3,axis = 0)
tranining_data_label = np.append(fold2_true_label,fold3_true_label,axis = 0)
test_data = fold1
#Q1
rows = training_data.shape[0]
test_data_rows = test_data.shape[0]
c1 = 0
c2 = 0
c3 = 0
for i in range(rows):
	if(tranining_data_label[i] == class_label[0]):
		c1+=1
	elif(tranining_data_label[i] == class_label[1]):
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
	if(tranining_data_label[i] == class_label[0]):
		class1[counter1] = training_data[i]
		counter1 += 1
	elif(tranining_data_label[i] == class_label[1]):
		class2[counter2] = training_data[i]
		counter2 += 1
	else:
		class3[counter3] = training_data[i]
		counter3 += 1

classes = list()
classes.append(class1)
classes.append(class2)
classes.append(class3)

for i in range(3):
	a_prior = round(classes[i].shape[0]/rows,2)
	mean = np.around(np.mean(classes[i],axis=0),decimals = 2)
	cov_matrix = np.around(np.cov(classes[i],rowvar = 0, bias = 1),decimals = 2)
	variances = cov_matrix.diagonal()
	cov = np.diag(variances)
	prior.append(a_prior)
	means.append(mean)
	covs.append(cov)
	# print(prior,file = f)
	# print(mean.tolist(),file = f)
	# print(cov.tolist(),file= f)

temp_label = list()
for i in range(test_data_rows):
	result = navie_bayes_classifier(test_data[i],means,covs,prior)
	temp_label.append(result)
	# print(test_data[i],end=",")
	# print(result)
result_label = np.array(temp_label)

c1 = 0
c2 = 0
c3 = 0
for i in range(test_data_rows):
	if(fold1_true_label[i] == class_label[0]):
		c1+=1
	elif(fold1_true_label[i] == class_label[1]):
		c2+=1
	else:
		c3+=1
class_static = list()
class_static.append(c1)
class_static.append(c2)
class_static.append(c3)

#compute the confusion matrix
confusion = np.zeros((3,3))
for i in range(test_data_rows):
	if(result_label[i] == class_label[0]):
			if(fold1_true_label[i] == class_label[0]):
				confusion[0][0] += 1
			elif(fold1_true_label[i] == class_label[1]):
				confusion[0][1] += 1
			elif(fold1_true_label[i] == class_label[2]):
				confusion[0][2] += 1
	elif(result_label[i] == class_label[1]):
			if(fold1_true_label[i] == class_label[0]):
				confusion[1][0] += 1
			elif(fold1_true_label[i] == class_label[1]):
				confusion[1][1] += 1
			elif(fold1_true_label[i] == class_label[2]):
				confusion[1][2] += 1
	elif(result_label[i] == class_label[2]):
			if(fold1_true_label[i] == class_label[0]):
				confusion[2][0] += 1
			elif(fold1_true_label[i] == class_label[1]):
				confusion[2][1] += 1
			elif(fold1_true_label[i] == class_label[2]):
				confusion[2][2] += 1
print("-----------------------------------------",file = f)
print("Fold 1: fold 1 is tested, fold 2 and fold 3 are training data.",file = f);
print("The confusion matrix when fold 1 is tested:",file = f)
print(confusion,file = f)
print("Classification Report:",file = f)
print("Class: accuracy|precision|recall|fscore",file = f)
fold1_acc_navie = compute_accuracy(confusion,class_static)
fold1_pr_navie = compute_Pr(confusion)
fold1_re_navie = compute_Re(confusion)
fold1_fscore_navie = compute_f_score(fold1_pr_navie,fold1_re_navie)
print("1: [%.2f %.2f %.2f %.2f]"%(fold1_acc_navie[0],fold1_pr_navie[0],fold1_re_navie[0],fold1_fscore_navie[0]),file = f)
print("2: [%.2f %.2f %.2f %.2f]"%(fold1_acc_navie[1],fold1_pr_navie[1],fold1_re_navie[1],fold1_fscore_navie[1]),file = f)
print("3: [%.2f %.2f %.2f %.2f]"%(fold1_acc_navie[2],fold1_pr_navie[2],fold1_re_navie[2],fold1_fscore_navie[2]),file = f)
avgacc1 = (fold1_acc_navie[0]+fold1_acc_navie[1]+fold1_acc_navie[2])/3
avgpr1 = (fold1_pr_navie[0]+fold1_pr_navie[1]+fold1_pr_navie[2])/3
avgre1 = (fold1_re_navie[0]+fold1_re_navie[1]+fold1_re_navie[2])/3
avgfscore1= (fold1_fscore_navie[0]+fold1_fscore_navie[1]+fold1_fscore_navie[2])/3
print("avg: [%.2f %.2f %.2f %.2f]"%(avgacc1,avgpr1,avgre1,avgfscore1),file = f)
print('\n',file = f)


print("-----------------------------------------",file = f)
print("Averaged value of evaluation over 3 folds:",file=f)
print("Class:accuracy|precision|recall|fscore",file = f)
avgacc_over = (avgacc1+avgacc2+avgacc3)/3
avgpr_over = (avgpr1+avgpr2+avgpr3)/3
avgre_over = (avgre1+avgre2+avgre3)/3
avgfs_over = (avgfscore1+avgfscore2+avgfscore3)/3
print("Avg: [%.2f %.2f %.2f %.2f]"%(avgacc_over,avgpr_over,avgre_over,avgfs_over),file = f)
print("-----------------------------------------",file = f)

print("Averaged value of evaluation in each class:",file=f)
print("Class:accuracy|precision|recall|fscore",file = f)
Avgacc_c1 = list()
for i in range(3):
	acc = (fold1_acc_navie[i]+fold2_acc_navie[i]+fold3_acc_navie[i])/3
	pr = (fold1_pr_navie[i]+fold2_pr_navie[i]+fold3_pr_navie[i])/3
	re = (fold1_re_navie[i]+fold2_re_navie[i]+fold3_re_navie[i])/3
	fs = (fold1_fscore_navie[i]+fold2_fscore_navie[i]+fold3_fscore_navie[i])/3
	print("Avg_c%d: [%.2f %.2f %.2f %.2f]"%((i+1),acc,pr,re,fs),file = f)
print("-----------------------------------------",file = f)

# i = 0
# avr_acc = (fold1_acc_navie+fold2_acc_navie+fold3_acc_navie)/3
# avr_pr = list()
# avr_re = list()
# avr_fscore = list()
# for i in range(confusion.shape[0]):
# 	pr = (fold1_pr_navie[i]+fold2_pr_navie[i]+fold3_pr_navie[i])/3
# 	re = (fold1_re_navie[i]+fold2_re_navie[i]+fold3_re_navie[i])/3
# 	fscore = (fold1_fscore_navie[i]+fold2_fscore_navie[i]+fold3_fscore_navie[i])/3
# 	avr_pr.append(pr)
# 	avr_re.append(re)
# 	avr_fscore.append(fscore)
# print("-----------------------------------------",file = f)
# print("The accuracy averaged over 3 folds is:",file = f)
# print(round(avr_acc,2),file = f)
# print("The precision for each class averaged over 3 folds is:",file = f)
# for i in range(confusion.shape[0]):
# 	print("%s    %.2f"%(class_label[i],round(avr_pr[i],2)),file = f)
# print("The recall for each class averaged over 3 folds is:",file = f)
# for i in range(confusion.shape[0]):
# 	print("%s    %.2f"%(class_label[i],round(avr_re[i],2)),file = f)
# print("The f-score for each class averaged over 3 folds is:",file = f)
# for i in range(confusion.shape[0]):
# 	print("%s    %.2f"%(class_label[i],round(avr_fscore[i],2)),file = f)
