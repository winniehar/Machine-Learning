import pandas as pd
from math import sqrt
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

def knn_euclid(row1, row2):
	distance = 0.0
	for i in range(len(row1)-1):
		distance = distance + (row1[i] - row2[i])**2
	return sqrt(distance)

def get_neighbors(dataset, query, k):
	distances = list()
	for i in range(len(dataset)):
		dist = knn_euclid(query, dataset[i])
		distances.append((i, dist))
	distances.sort(key=lambda tup: tup[1])
	neighbors = list()
	for j in range(k):
		neighbors.append(distances[j][0])
	
	return neighbors

def make_prediction(dataset, query, k):
	neighbors = get_neighbors(dataset, query, k)
	output = list()
	for i in range(len(neighbors)):
		output.append(dataset[neighbors[i]][-1])
	pred = max(set(output), key=output.count)
	return pred

dataset = [[3,0,0,0,0,0,0,1],[1,2,1,1,1,0,0,1],[0,0,1,1,1,0,0,1],
	[0,0,1,0,3,1,1,0],[0,1,0,0,0,1,1,0]]

query = [0,1,1,0,0,1,1]
query_1 = np.array(query).reshape(1, -1)


#Question 7)1)
classification = make_prediction(dataset, query, 1)
print("Nearest neighbor will classify it as ", classification)

#Question 7)2)
classification = make_prediction(dataset, query, 3)
print("3 nearest neighbors will classify it as ", classification)

df = pd.DataFrame(dataset, columns = ['Money','Free','For','Gambling','Fun','Machine','Learning','Spam'])

x = df.drop(columns=['Spam'])
x = np.array(x)
y = df["Spam"].values

def knn(x, y, query, k, method):
	if method == 'euclid':
		neigh = KNeighborsClassifier(n_neighbors=k, p=2)
		neigh.fit(x,y)
		prediction = neigh.predict(query)
	elif method == 'cosine':
		neigh = KNeighborsClassifier(n_neighbors=k, metric="cosine")
		neigh.fit(x,y)
		prediction = neigh.predict(query)
	elif method == 'weighted':
		neigh = KNeighborsClassifier(n_neighbors=k, weights="distance", metric="euclidean")
		neigh.fit(x,y)
		prediction = neigh.predict(query)
	return prediction

#Alternative solution for 7)1)
prediction = knn(x,y,query_1,1,'euclid')
print("Prediction using 1-NN and Euclidean distance is : ", prediction)

#Alternative solution for 7)2)
prediction = knn(x,y,query_1,3,'euclid')
print("Prediction using 3-NNs and Euclidean distance is : ", prediction)

#Question 7)3)
prediction = knn(x,y,query_1,5,'weighted')
print("Prediction using 5-NNs and weighted k-NN is : ", prediction)

#Question 7)4)
prediction = knn(x,y,query_1,3,'cosine')
print("Prediction using 3-NNs and cosine similarity is : ", prediction)

'''
for i in neighbors:
	output = training[i][-1]

for i in range(len(neighbors)):
	print(dataset[neighbors[i]][-1])
	output.append(dataset[neighbors[i]][-1])
'''