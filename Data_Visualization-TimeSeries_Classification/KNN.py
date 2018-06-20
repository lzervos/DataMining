from sklearn.base import BaseEstimator
from collections import defaultdict
from lcs import lat_lng
from fastdtw import fastdtw
from haversine import haversine
import numpy as np

class KNN(BaseEstimator):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors

    def fit(self,data,target):
        self.X = data
        self.y = target
        return self

    def _distance(self,instance1,instance2):
        #This is for manhattan distance
        #return sum(abs(instance1 - instance2))  
        #This is for euclidean distance which is a little better
        #return np.sqrt(sum((instance1 - instance2)**2))
        inst1=np.array(lat_lng(instance1))
        inst2=np.array(lat_lng(instance2))
        distance,path=fastdtw(inst1,inst2,dist=haversine)
        return distance

    def _instance_prediction(self, test):
        #Sorted the instances by the distance
        distances = sorted((self._distance(x, test), y) for x, y in zip(self.X, self.y))
        #Taking the instances for each class from the k nearest
        instances= self._classes_instances(distances[:self.n_neighbors])
        instances_by_class = defaultdict(list)
        for d, c in instances:
            instances_by_class[c].append(d)
        counts = [(sum(value),key) for key,value in instances_by_class.iteritems()]
        #Find the class with the most instances
        majority = max(counts)
        return majority[1]

    def _classes_instances(self, distances):
	    cl_inst=[(1,y) for x,y in distances]
	    return cl_inst
        #cl_inst = [(1,y) for x,y in distances if x == 0]
        #return cl_inst if cl_inst else [(1/x, y) for x, y in distances] #After we have found the k n$

    def predict(self, X):
        #Calling instance prediction for each instance
        return [self._instance_prediction(x) for x in X]

    def score(self, X, y):
        return sum(1 for p, t in zip(self.predict(X), y) if p == t) / len(y)
