import numpy.matlib
import numpy.linalg
import numpy as np
import scipy.stats
import collections
from collections import Counter
import sys

# a = np.array([(1,5,6,3),(1,5,4,3),(3,9,5,8)])
# print(a)
# lista = a.tolist()
# for i in range(3):
# 	print("max: ",max(lista))
# 	print("index: ", lista.index(max(lista)))

# k=3
# lists = [[] for _ in range(k)]
# a = np.array([1,2,3])
# # lista = a.tolist()
# b = np.array([4,5,6])
# # listb = b.tolist()
# lists[0].append(a)
# lists[0].append(b)
# print(lists)
# print(lists[0][0])

# a = np.array([(1,5,6,3),(1,5,4,3),(3,9,5,8)])
# b = np.array([a]*3)
# print(b[0])

# a = np.empty([0,4])
# b = np.array([1,2,3,4])
# d = np.array([4,5,6,7])
# c = np.append(a,b)
# f = np.append(c,d,axis = 0)
# print(f)
# a = np.array([1,2,3,4])
# b = np.array([4,5,6,7])
# c = np.array([8,9,10,11])
# lists = [[] for _ in range(3)]
# lists[0].append(a.tolist())
# lists[0].append(b.tolist())
# lists[0].append(c.tolist())
# lists[1].append(b.tolist())
# lists[2].append(c.tolist())
# print(lists)
# array = np.array(lists[0])
# print(array)

# fi = np.zeros([1,3])
# print(fi)
# fi[0] = array
# print(fi)


# a = np.arange(15).reshape(3,5)
# print(a)
# print("----")
# b =a.sum(axis = 1)
# # c = b.T
# # print(c.reshape(-1,1))
# a = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],[50,60,70]])
# print(a)
# b = np.array([[1],[2],[3],[4],[5]])
# print(b)
# print(a/b)

# b = np.zeros((3,4))
# a = np.array([(1,2),(0,1),(3,4),(7,8)])
# print(type(a))
# norms = list()
# for i in range(a.shape[0]):
# 	norm = numpy.linalg.norm(a[i])
# 	norms.append(norm)
# print(type(np.argsort(norms)))
# c = np.array(norms)
# print(norms)
# print(c)
# print(a)
true_assignment = np.loadtxt("iris.data",delimiter = ',', dtype = str, usecols = np.arange(4,5))
print(collections.Counter(true_assignment))
label = collections.Counter(true_assignment)
for j in label:
	print(j)