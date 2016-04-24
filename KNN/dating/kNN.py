from  numpy import *
import operator
import matplotlib
import matplotlib.pyplot as plt

'''
def createDataSet():
	# use function array() to create a array
	group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
	labels = ['A','A','B','B']
	return group, labels
'''

def classify0(inX, dataSet, labels, k):
	# shape will get the dimensions of list.
	dataSetSize = dataSet.shape[0] 
	# tile will repeat the list for several times in row and row
	diffMat = tile(inX,(dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
    # calc sum in row 
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort() # get the index of args list by asc order
	classCount = {}
	for i in range(k):
		votelabel = labels[sortedDistIndicies[i]]
		classCount[votelabel] = classCount.get(votelabel,0)+1
	sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def file2matrix(fileName):
	fr = open(fileName)
	arrayOLines = fr.readlines()
	numberofLines = len(arrayOLines)
	returnMat = zeros((numberofLines,3)) # create a zero marix of numberofLines rows and 3 cols
	classLabelVector = []
	index = 0
	for line in arrayOLines:
		line = line.strip()
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]
		classLabelVector.append(int(listFromLine[-1]))
		index += 1
	return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingCalssTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with %d, the real answer is: %d"%(classifierResult,datingLabels[i])
        if(classifierResult!=datingLabels[i]):
            errorCount+=1.0
    print "the total error rate is %f"%(errorCount/float(numTestVecs))

def classfyPerson():
	resultList = ['not at all','in small dose','in large dose']
	percentTats = float(raw_input("percentage of time spent playting video games?"))
	ffMiles = float(raw_input("frequent filer miles earned per year?"))
	iceCream = float(raw_input("liters of ice crean consumed per year?"))
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat,ranges,minVals = autoNorm(datingDataMat)
	inArr = array([ffMiles,percentTats,iceCream])
	classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
	print "You will probably like this person: ",resultList[classifierResult-1]



'''
fig = plt.figure()
ax = fig.add_subplot(111) # draw picture of 1 row and 1 col as the 1st sub_picture
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*array(datingLabels),15.0*array(datingLabels))
plt.show()
'''

#datingCalssTest()
classfyPerson()
