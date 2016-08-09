from numpy import *
from numpy import linalg as la

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [1, 1, 1, 0, 0],
           [5, 5, 5, 0, 0]]

def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

def eulidSim(inA,inB):
    return 1.0/(1.0 + la.norm(inA - inB))

def pearSim(inA,inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5*corrcoef(inA,inB,rowvar=0)[0][1]

def cosSim(inA,inB):
    num = float(inA.T * inB)
    denorm = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5*(num/denorm)

def standEst(dataMat,user,simMeas,item):# recomend based on similar user
    n = shape(dataMat)[1] # n things
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0:  # skip unrated items
            continue
        overLap = nonzero(logical_and(dataMat[:,item].A>0,dataMat[:,j].A>0))[0] # items index who both rated item and j by same people
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap,item],dataMat[overLap,j])
        #print('the %d and %d similarity is :%f '%(item,j,similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal/simTotal

def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = nonzero(dataMat[user,:].A==0)[1] #find unrated items
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        itemScores.append((item, estimatedScore))
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]

def testRecommend():
    myMat = matrix([[4, 4, 0, 2, 2],
        [4, 0, 0, 3, 3],
        [4, 0, 0, 1, 1],
        [1, 1, 1, 2, 0],
        [2, 2, 2, 0, 0],
        [5, 5, 5, 0, 0],
        [1, 1, 1, 0, 0]])
    print(recommend(myMat,2,simMeas= pearSim))

def svdEst(dataMat,user,simMeas,item):
    n = dataMat.shape[0]
    simTotal = 0.0
    ratSimTotal = 0.0
    U,Sigma,VT = la.svd(dataMat) # U m*m matix VT n*n Matrix
    Sig4 = mat(eye(4)*Sigma[:4])
    xformedItems = dataMat.T * U[:,:4]*Sig4.I # M = U*S*V.T ---> V=M.T*U*S.I   U is the orthogonal matrix
    # xformedItems[i,:] is the i-th component in low dimension
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating==0 or j==item:
            continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print('the %d and %d similarity is :%f ' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity*userRating
    if simTotal ==0:
        return 0
    else:
        return ratSimTotal/simTotal


def testSVD():
    myMat = mat(loadExData2())
    print(recommend(myMat,1,estMethod=svdEst))

testSVD()