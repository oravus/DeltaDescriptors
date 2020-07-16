import matplotlib.pyplot as plt

def saveDistMat(dMat,mInds,outPath):
    plt.imshow(dMat)
    plt.colorbar()
    plt.plot(mInds,'.',c='tab:orange')
    plt.savefig(outPath)
    plt.close()
