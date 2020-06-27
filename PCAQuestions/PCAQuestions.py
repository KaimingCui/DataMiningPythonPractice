import numpy.matlib
import numpy.linalg
import numpy as np
import sys

# read the data from file and stored as numpy array
filename = sys.argv[1]
data = np.loadtxt(filename,delimiter = ',',usecols = np.arange(0,10))
f = open("assign2_iuKaimingCui.txt",'w+')


#Question a: Write a script to apply z-normalization to this dataset. For the reamining questions, you should use z-normalized dataset instead of the original dataset.
mean = np.mean(data,axis=0)
std = np.std(data,axis=0)
z_nomalization = (data-mean)/std
print("Question a:")
print("Question a:",file=f)
print("z-score normalization: ")
print("z-score normalization: ",file=f)
print(z_nomalization)
print(z_nomalization,file=f)
print("-----------------------------------------------------")
print("-----------------------------------------------------",file=f)
print('\n')
print('\n',file=f)
rows = data.shape[0]
cols = data.shape[1]

#Question b: calculate the sample covariance matrix
cov = np.matlib.zeros((10,10))
i=0
for i in range(rows):
	cov += np.outer(z_nomalization[i],z_nomalization[i])
covar = cov/rows
print("Question b:")
print("Question b:",file=f)
print("sample covariance matrix calculated by using formula: ")
print("sample covariance matrix calculated by using formula: ",file=f)
print(covar)
print(covar,file=f)
print('\n')
covar_verify = np.cov(z_nomalization,rowvar=0,bias = 1)
print('\n',file=f)
print("verify by using np.cov function: ")
print("verify by using np.cov function: ",file=f)
print(covar_verify)
print(covar_verify,file=f)
print("-----------------------------------------------------")
print("-----------------------------------------------------",file=f)
print('\n')
print('\n',file=f)
	
#Question c: power iteration method to get eigenvector and eigenvalue
x0 = np.ones(shape=(cols,1))
x1 = np.zeros(shape = (cols,1))
threshold = 0.0000001
difference = 1
x_index = 0
while difference > threshold:
	x_index += 1
	x1 = np.dot(covar,x0)
	max_absolute = np.abs(x1).max()
	j=0
	for j in range(cols):
		if x1[j] == max_absolute or x1[j] == -1*max_absolute:
			x1 = x1/x1[j]
	difference = np.linalg.norm(x1-x0)
	x_previous = x0
	x0 = x1
# largest_eig_value = x1.max()/x_previous.max()
largest_eig_value = max_absolute
final_eig_vector = x1/np.linalg.norm(x1)
print("Question c threshold set 0.00000001:")
print("Question c threshold set 0.00000001:",file=f)
print("eigen value by power iteration: ")
print("eigen value by power iteration: ",file=f)
print(largest_eig_value)
print(largest_eig_value,file=f)
print("eigen vector by power iteration: ")
print("eigen vector by power iteration: ",file=f)
print(final_eig_vector)
print(final_eig_vector,file=f)
print('\n')
print('\n',file=f)
eig_value,eig_vector = np.linalg.eig(covar)
eig_value_index = np.argsort(-eig_value)
n_index = eig_value_index[0:1]
print("verify eigen value by using linalg.eig function: ")
print("verify eigen value by using linalg.eig function: ",file=f)
print(eig_value[n_index])
print(eig_value[n_index],file=f)
print("verify eigen vector by using linalg.eig function: ")
print("verify eigen vector by using linalg.eig function: ",file=f)
print(eig_vector[:,n_index[0]])
print(eig_vector[:,n_index[0]],file=f)
print("-----------------------------------------------------")
print("-----------------------------------------------------",file=f)
print('\n')
print('\n',file=f)



#Question d: Use linalg.eig to find first two dominant eigenvectors of covariance matrix
eig_value,eig_vector = np.linalg.eig(covar)
# print("verify eigen value: ",eig_value)
# print("verify eigen vector: ",eig_vector)
n_index = eig_value_index[0:2]
first_two_eigvector = eig_vector[:,n_index]
first_two_eigvalue = eig_value[n_index]
reduct_dimension = np.dot(first_two_eigvector.T,z_nomalization.T)
# variance = np.var(reduct_dimension,axis = 1)
variance = np.sum(np.trace(np.cov(reduct_dimension)))
print("Question d")
print("Question d",file=f)
print("The total variance of projected data is: ")
print("The total variance of projected data is: ",file=f)
print(variance)
print(variance,file=f)
print("-----------------------------------------------------")
print("-----------------------------------------------------",file=f)
print('\n')
print('\n',file=f)
# print("using first two eigen vector to reduct the dimensionality of original data: ")
# print("note: each column is a data point")
# print(reduct_dimension)

#Question e: Use linalg.eig to find all the eigenvectors
covar = np.cov(z_nomalization,rowvar=0,bias = 1)
eig_value,eig_vector = np.linalg.eig(covar)
eigen_value_index = np.argsort(-eig_value)
sorted_eigen_value = eig_value[eigen_value_index]     #sort the eigen values and also rearange the corresponding vectors
sorted_eigen_vector = eig_vector[:,eigen_value_index]
A = np.diag(sorted_eigen_value)
print("Question e")
print("Question e",file=f)
print("Covariance matrix: ")
print(covar)
print('\n')
print("Covariance matrix: ",file = f)
print(covar,file = f)
print('\n',file = f)
print("U: ")
print(sorted_eigen_vector)
print('\n')
print("A: ")
print(A)
print('\n')
print("UT: ")
print(sorted_eigen_vector.T)
print("U: ",file=f)
print(sorted_eigen_vector,file=f)
print('\n',file=f)
print("A: ",file=f)
print(A,file=f)
print('\n',file=f)
print("UT: ",file=f)
print(sorted_eigen_vector.T,file=f)
print("-----------------------------------------------------")
print("-----------------------------------------------------",file=f)
print('\n')
print('\n',file=f)


#Question f: PCA
total_var_org = np.sum(eig_value)
total_var = 0
number = 0
sort_eigvalue = eig_value[np.argsort(-eig_value)]  #from large to small
for i in  sort_eigvalue:
	total_var += i
	number += 1
	if total_var/total_var_org >= 0.95:
		break;
eigen_value_index = np.argsort(-eig_value)
n_eigen_value_index = eigen_value_index[0:number]
n_eigen_vector = eig_vector[:,n_eigen_value_index]
i = 0
print("Question f: ")
print("Question f: ",file=f)
print("The principal vectors we need to preserve 95% (each clomun is a principal vector): ")
print("The principal vectors we need to preserve 95% (each cloumn is a principal vector): ",file=f)
print(n_eigen_vector)
print(n_eigen_vector,file=f)
print('\n')
print('\n',file=f)
print("The first 10 data points co-ordinate (each row is a data point): ")
print("The first 10 data points co-ordinate (each row is a data point): ",file=f)
data_point = np.zeros(shape=(7,10))
for i in range(10):
	data_point[:,i] = np.dot(n_eigen_vector.T,z_nomalization[i].T) #data_point each column is a data point
print(data_point.T)
print(data_point.T,file=f)
print("-----------------------------------------------------")
print("-----------------------------------------------------",file=f)
print('\n')
print('\n',file=f)

#Question g
threshold = 0.001
projected_data = np.dot(n_eigen_vector.T,z_nomalization.T)
covar_projected = np.cov(projected_data)
n_largest_eig_value = eig_value[n_eigen_value_index]
sum_eig_value = np.sum(n_largest_eig_value)
covar_project_data = np.sum(np.trace(covar_projected))
difference = np.abs(sum_eig_value - covar_project_data)
print("Question g: ")
print("Question g: ",file=f)
print("The co-variance matrix of the projected data points: ")
print(covar_projected)
print('\n')
print("Then sum of its diagonal: ")
print(covar_project_data)
print('\n')
print("The sum of eigen value corresponding to principal vectors on which the data is projected: ")
print(sum_eig_value)
print('\n')
print("The co-variance matrix of the projected data points: ",file=f)
print(covar_projected,file=f)
print('\n',file=f)
print("Then sum of its diagonal: ",file=f)
print(covar_project_data,file=f)
print('\n',file=f)
print("The sum of eigen value corresponding to principal vectors on which the data is projected: ",file=f)
print(sum_eig_value,file=f)
print('\n',file=f)
if difference < threshold:
	print("They are matched. Using threshold 0.001")
	print("They are matched. Using threshold 0.001",file=f)
else:
	print("Thery are not matched. Using threshold 0.001")
	print("Thery are not matched. Using threshold 0.001",file=f)
print("-----------------------------------------------------")
print("-----------------------------------------------------",file=f)



