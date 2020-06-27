import numpy.matlib
import numpy.linalg
import numpy as np
import scipy.stats 
from scipy.stats import multivariate_normal
import collections
from collections import Counter
import sys



#Update P(Ci|xi) that is W
def updata_W(data,means,covars,P,num_points,k):
	i = 0
	P_multi_pdf = np.zeros((num_points,k))
	Sum_P_multi_pdf = np.zeros((num_points,1))
	for i in range(k):
		P_multi_pdf[:,i] = P[i]*multivariate_normal.pdf(data,means[i],covars[i])
	Sum_P_multi_pdf = P_multi_pdf.sum(axis = 1).reshape(-1,1)
	W = P_multi_pdf/Sum_P_multi_pdf
	return W

#update P(Ci)
def update_P(W):
	P = W.sum(axis=0)/W.sum() #/n
	return P

#update Mean for each cluster using the weight W
def update_Mean(data,W,k):
	means = np.zeros((k,4)) #4 dimension
	i = 0
	for i in range(k):
		#weight of each cluster
		weight = W[:,i] 
		means[i] = np.average(data,axis=0,weights = weight)
	return means

#update var and covar for each cluster using the weight W
def update_covar(data,W,means,k):
	i = 0
	covars = np.array([np.zeros((4,4))]*k)
	variances = np.zeros((k,4))
	for i in range(k):
		weight = W[:,i]
		variances[i] = np.average(np.square(data - means[i]), axis = 0, weights = weight)
		covars[i] = np.diag(variances[i])
	return covars


# K = 3
# cov = np.array([np.eye(4)] * K)
# print(cov)
# read the dataset only the first four attributes
# axis=0 按列
# filename = sys.argv[1]
# k = sys.argv[2]
# data = np.loadtxt(filename,delimiter = ',',usecols = np.arange(0,4))
data = np.loadtxt("iris.data",delimiter = ',',usecols = np.arange(0,4))
k = 3
f = open("Cui-assign2.pdf",'w+')
convergence = 0.00001

total_points = data.shape[0]
dimension = data.shape[1]
n_points_cluster = int(total_points/k)
n_points = total_points
clusters = np.array([np.zeros((n_points_cluster,4))]*k)
means = np.zeros((k,4))
covars = np.array([np.zeros((4,4))]*k)

i = 0
begin_point = 0
for i in range(k):
	cluster = data[begin_point:begin_point+n_points_cluster,:]
	begin_point = (i+1)*n_points_cluster
	clusters[i] = cluster
	means[i] = np.mean(cluster,axis = 0)
	covars[i] = np.cov(cluster,rowvar = 0, bias = 1) #bais = 1 means: divided by N instead of N-1
	# variances.append(np.var(cluster,axis = 0))

W = np.ones((n_points,k))/k #(P(Ci|xi)) n by k matrix
P = W.sum(axis = 0)/W.sum()   #P(Ci) for each cluster
# print(W)
# print("---")
# print(P)

#iteration util convergence reached
sum_dist = 1
old_means = np.zeros((k,4))
iteration = 0
while sum_dist > convergence:
	iteration  += 1
	old_means = means
	W = updata_W(data,means,covars,P,total_points,k)
	P = update_P(W)
	means = update_Mean(data,W,k)
	covars = update_covar(data,W,means,k)
	j = 0
	sum_dist = 0
	for j in range(k):
		sum_dist += numpy.linalg.norm(means[j] - old_means[j])

#accoding to W / P(Ci|xi) to assign the points to clusters
final_clusters = [[] for _ in range(k)]
i = 0
result = list()
for i in range(n_points):
	point = W[i,:]
	list_point = point.tolist()
	which_cluster = list_point.index(max(list_point))
	if(which_cluster == 0):
		result.append("Iris-setosa")
	elif(which_cluster == 1):
		result.append("Iris-versicolor")
	else:
		result.append("Iris-virginica")
	final_clusters[which_cluster].append(data[i])

#output
print("Mean:")
i = 0
norms = list()
for i in range(k):
	norm = numpy.linalg.norm(means[i])
	norms.append(norm)
index_incre = np.argsort(norms)
i = 0
for i in range(k):
	print(np.around(means[index_incre[i]],decimals=3))


print('\n')
print("-------------------------")
print("Covariance Matrix:")
final_covars = np.array([np.zeros((4,4))]*k)
for i in range(k):
	final_covars[i] = np.cov(np.array(final_clusters[i]),rowvar = 0, bias = 1)
for i in range(k):
	print(np.around(final_covars[index_incre[i]], decimals = 3))
	print("######")
	print(np.around(covars[index_incre[i]],decimals = 3))
	print('\n')


print("-------------------------")
print("Iteration count = ",iteration)
print('\n')


print("-------------------------")
print("Cluster Membership:")
for i in range(k):
	# print(len(final_clusters[index_incre[i]]))
	# print(np.array(final_clusters[index_incre[i]]))
	for j in range(len(final_clusters[index_incre[i]])):
		print(np.array(final_clusters[index_incre[i]][j]),end = ",")
	print('\n')

print("-------------------------")
print("Size:")
for i in  range(k):
	print(len(final_clusters[index_incre[i]]))

print('\n')
print("-------------------------")
print("Purity:")
true_assignment = np.loadtxt("iris.data",delimiter = ',', dtype = str, usecols = np.arange(4,5))
true_assignment_list = true_assignment.tolist()
EM_cluster_label = collections.Counter(result)
true_label = collections.Counter(true_assignment_list)
# print(EM_cluster_label)
# print(true_label)
counter = 0
clusters_purity = list()
for k in EM_cluster_label:
	temp_list = list()
	for t in true_label:
		counter = 0
		for i in range(n_points):
			if result[i] == k and true_assignment_list[i] == t:
				counter += 1
		temp_list.append(counter)
	temp_max = max(temp_list)
	clusters_purity.append(temp_max)
purity = sum(clusters_purity)/n_points
print(purity)
# counter = 0
# for i in  range(n_points):
# 	if result[i] == true_assignment_list[i]:
# 		counter += 1
# print(counter/n_points)
