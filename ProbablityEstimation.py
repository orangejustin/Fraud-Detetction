import numpy as np
import math
import operator
import copy
import tensorflow as tf

'''
Simple Class used for generate probability estimation
of given number of sample point
The generated data will be used to feed neuron network
'''
class ProbabilityEstimation:
    '''
    dataSet.shape = [sampleNumber, variableNumber]
    '''
    def __init__(self, dataset, clusterNumber):
        self.dataset = dataset
        self.Kmus = np.zeros((clusterNumber, dataset.shape[1]))
        randInds = np.random.permutation(dataset.shape[0])
        self.Kmus = dataset[randInds[0:clusterNumber],:]

    def setClusterNumber(self, clusterNumber):
        self.Kmus = np.zeros((clusterNumber, self.dataset.shape[1]))
        randInds = np.random.permutation(self.dataset.shape[0])
        self.Kmus = self.dataset[randInds[0:clusterNumber],:]

    def calcSqDistance(self):
        return ((-2 * self.dataset.dot(self.Kmus.T) + np.sum(np.multiply(self.Kmus,self.Kmus), axis=1).T).T + np.sum(np.multiply(self.dataset,self.dataset), axis=1)).T

    def determineRnk(self, sqDmat):
        m = np.argmin(sqDmat, axis=1)
        return np.eye(sqDmat.shape[1])[m]

    def recalcMus(self, Rnk):
        RnkSum = np.sum(Rnk, axis=0)
        indexOfZero = np.where(RnkSum == 0)
        safeRnk = np.delete(Rnk, indexOfZero, axis = 1)
        self.Kmus = np.delete(self.Kmus, indexOfZero, axis = 0)
        Kmus = np.zeros((len(safeRnk[0]), self.dataset.shape[1]))
        return (np.divide(self.dataset.T.dot(safeRnk), np.sum(safeRnk, axis=0))).T

    def run(self, maxIteration = 1000):
        for iter in range(maxIteration):
            sqDmat = self.calcSqDistance()
            Rnk = self.determineRnk(sqDmat)
            KmusNew = self.recalcMus(Rnk)
            KmusOld = self.Kmus
            self.Kmus = KmusNew
            if np.sum(np.abs(KmusOld.reshape((-1,1)) - self.Kmus.reshape((-1,1)))) < 1e-6:
                break

    '''
    return two numpy array:
    Kmus with shape [clusterNumber, variableNumber]
    targetProbability with shape [clusterNumber]
    '''
    def getDataSets(self, maxDistance=0.1):
        sqDmat = self.calcSqDistance()
        Rnk = self.determineRnk(sqDmat)
        for i in range(len(Rnk)):
            j = np.argmax(Rnk[i])
            if sqDmat[i][j] > maxDistance:
                Rnk[i][j] = 0
        Rnk = np.sum(Rnk, axis=0)
        return self.Kmus, Rnk/self.dataset.shape[0]