import matplotlib.pyplot as plt
import numpy as np

def saveDistMat(dMat,mInds,outPath):
    plt.imshow(dMat)
    plt.colorbar()
    plt.plot(mInds,'.',c='tab:orange')
    plt.savefig(outPath)
    plt.close()
    return

def getPR(mInds,gt,locRad):

    positives = np.argwhere(mInds!=-1)[:,0]
    tp = np.sum(gt[positives] <= locRad)
    fp = len(positives) - tp

    negatives = np.argwhere(mInds==-1)[:,0]
    tn = np.sum(gt[negatives]>locRad)
    fn = len(negatives) - tn

    assert(tp+tn+fp+fn==len(gt))

    if tp == 0:
        return 0,0,0 # what else?

    prec = tp/float(tp+fp)
    recall = tp/float(tp+fn)
    fscore = 2*prec*recall/(prec+recall)

    return prec, recall, fscore

def getPRCurve(mInds,mDists,gt,locRad):

    prfData = []
    lb, ub = mDists.min(),mDists.max()
    step = (ub-lb)/100.0
    for thresh in np.arange(lb,ub+step,step):
        matchFlags = mDists<=thresh
        outVals = mInds.copy()
        outVals[~matchFlags] = -1

        p,r,f = getPR(outVals,gt,locRad)
        prfData.append([p,r,f])
    return np.array(prfData)

def getPAt100R(dists,maxLocRad):
    pAt100R = []
    for i1 in range(maxLocRad):
        pAt100R.append([np.sum(dists<=i1)])
    pAt100R = np.array(pAt100R) / float(len(dists))
    return pAt100R

def evaluate(resPath, insPath1=None, insPath2=None, maxLocRad=50):
    res = np.load(resPath)
    mInds = res["matchInds"]
    mDists = res["matchDists"]

    if insPath1 is None or insPath2 is None:
        # assume 1-to-1 frame correspondence with frame indices as co-ords
        rCords = np.arange(len(mInds))
        qCords = np.arange(len(mInds))
        mrCords = rCords[mInds]
        dists = abs(qCords - mrCords)
    else:
        pass #TODO

    pAt100R = getPAt100R(dists,maxLocRad)

    print(pAt100R[[1,5,10,20]])
    return pAt100R