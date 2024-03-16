"""*****************************************************************************************
IIIT Delhi License
Copyright (c) 2023 Supratim Shit
*****************************************************************************************"""

from sklearn.datasets import fetch_kddcup99
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import wkpp as wkpp 
import numpy as np
import random


# Real data input
dataset = fetch_kddcup99()								# Fetch kddcup99 
data = dataset.data										# Load data
data = np.delete(data,[0,1,2,3],1) 						# Preprocess
data = data.astype(float)								# Preprocess
data = StandardScaler().fit_transform(data)				# Preprocess

n = np.size(data,0)										# Number of points in the dataset
d = np.size(data,1)										# Number of dimension/features in the dataset.
k = 17													# Number of clusters (say k = 17)
Sample_size = 100										# Desired coreset size (say m = 100)

def D2(data, k):
    # Initialize B with one random point from data
    B = [data[np.random.choice(range(n))]]
    for _ in range(1, k):
        print("Current k is ", k)
        # Calculate the distance from each point in data to the set B
        distances = np.min(cdist(data, B, 'euclidean')**2, axis=1)
        print("Distances",distances)
        # Calculate the probabilities for each point
        probabilities = distances / np.sum(distances)
        print("Probabilities",probabilities)
        # Sample a new point based on the calculated probabilities
        B.append(data[np.random.choice(range(n), p=probabilities)])
        print("B",B)
    return np.array(B)
	
centers = D2(data,k)									# Call D2-Sampling (D2())
def Sampling(data, k, centers, Sample_size):
    # Initialize arrays to store distances and weights
    distances = np.zeros(len(data))
    weights = np.zeros(len(data))
    
    # First loop: Calculate the distance from each point in data to the nearest center
    for i in range(len(data)):
        min_distance = float('inf')
        for j in range(len(centers)):
            distance = np.linalg.norm(data[i] - centers[j])**2
            if distance < min_distance:
                min_distance = distance
        distances[i] = min_distance
    
    phi = np.sum(distances) / len(data)
    
    # Second loop: Calculate the sampling probabilities for each point
    probabilities = np.zeros(len(data))
    
    for i in range(len(distances)):
        probabilities[i] = distances[i] / phi
    
    # Normalize the probabilities
    probabilities /= np.sum(probabilities)
    
    # Third loop: Sample points based on calculated probabilities and calculate their weights 
    coreset_indices = []
    coreset_weights = []
    
    for _ in range(Sample_size):
        index = np.random.choice(len(data), p=probabilities)
        coreset_indices.append(index)
        weight = 1 / (Sample_size * probabilities[index])
        weights.append(weight)

    coreset = data[coreset_indices]
    
    return coreset, weights


coreset, weight = Sampling(data,k,centers,Sample_size)	# Call coreset construction algorithm (Sampling())

#---Running KMean Clustering---#
fkmeans = KMeans(n_clusters=k,init='k-means++')
fkmeans.fit_predict(data)

#----Practical Coresets performance----# 	
Coreset_centers, _ = wkpp.kmeans_plusplus_w(coreset, k, w=weight, n_local_trials=100)						# Run weighted kMeans++ on coreset points
wt_kmeansclus = KMeans(n_clusters=k, init=Coreset_centers, max_iter=10).fit(coreset,sample_weight = weight)	# Run weighted KMeans on the coreset, using the inital centers from the above line.
Coreset_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
coreset_cost = np.sum(np.min(cdist(data,Coreset_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_practicalCoreset = abs(coreset_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from practical coreset, here fkmeans.inertia_ is the optimal cost on the complete data.

#-----Uniform Sampling based Coreset-----#
tmp = np.random.choice(range(n),size=Sample_size,replace=False)		
sample = data[tmp][:]																						# Uniform sampling
sweight = n*np.ones(Sample_size)/Sample_size 																# Maintain appropriate weight
sweight = sweight/np.sum(sweight)																			# Normalize weight to define a distribution

#-----Uniform Samling based Coreset performance-----# 	
wt_kmeansclus = KMeans(n_clusters=k, init='k-means++', max_iter=10).fit(sample,sample_weight = sweight)		# Run KMeans on the random coreset
Uniform_centers = wt_kmeansclus.cluster_centers_															# Compute cluster centers
uniform_cost = np.sum(np.min(cdist(data,Uniform_centers)**2,axis=1))										# Compute clustering cost from the above centers
reative_error_unifromCoreset = abs(uniform_cost - fkmeans.inertia_)/fkmeans.inertia_						# Computing relative error from random coreset, here fkmeans.inertia_ is the optimal cost on the full data.
	

print("Relative error from Practical Coreset is",reative_error_practicalCoreset)
print("Relative error from Uniformly random Coreset is",reative_error_unifromCoreset)
