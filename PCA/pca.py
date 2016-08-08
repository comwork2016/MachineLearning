from numpy import *
import matplotlib
import matplotlib.pyplot as plt

def loadDataSet(filename,delim='\t'):
    fr = open(filename)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [map(float,line) for line in stringArr] # map line to float type
    return mat(datArr)

def pca(dataMat,topNfeat = 9999999):
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar=0) #get the convariance matrix,each row represents a observations, with variable in the columns
    eigVals,eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1] # top N dimension
    redEigVects = eigVects[:,eigValInd] # get N eigvectors
    lowDDataMat = meanRemoved * redEigVects # liner algebra to get low dimension matrix
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # reconstruct data to high dimension to debug
    return lowDDataMat,reconMat

def showScatter(dataMat,reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:,0].flatten().A[0],dataMat[:,1].flatten().A[0],marker='^',s=9)
    ax.scatter(reconMat[:,0].flatten().A[0],reconMat[:,1].flatten().A[0],marker='o',s=50,c='red')
    plt.show()

dataMat = loadDataSet('testSet.txt')
lowDMat,reconMat = pca(dataMat,1)
showScatter(dataMat,reconMat)
print(shape(dataMat))
print(shape(lowDMat))
print(shape(reconMat))

def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = datMat.shape[1]
    for i in range(numFeat):
        meanVal = mean(datMat[nonzero(~isnan(datMat[:,i].A))[0],i])
        datMat[nonzero(isnan(datMat[:,i].A))[0],i] = meanVal
    return datMat

def showEigValueOfSecomData():
    dataMat = replaceNanWithMean()
    meanVals = mean(dataMat,axis=0)
    meanRemoved = dataMat - meanVals
    covMat = cov(meanRemoved,rowvar=0)
    eigVals,eigVects = linalg.eig(mat(covMat))
    print(eigVals)