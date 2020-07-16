import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm

def transformPCAEx(ftIn,pca):
    """
    Explicit transformation to optionally disable data centering
    """
    pm = pca.mean_
    pc = pca.components_
    diff = ftIn - pm[None,:]
    tfm = np.dot(diff,pc.transpose())
    return tfm
        
def performOps(data,opName,winL=None,pcaComp=None):

    if opName == "pca":
        f1, f2 = data
        ft1, ft2 = f1.copy(), f2.copy()
        if pcaComp is None:
            pca = PCA()
        else:
            pca = PCA(pcaComp)
      
        ft1P = pca.fit_transform(ft1)
        ft2P = transformPCAEx(ft2,pca)#pca.transform(ft2)
        return ft1P, ft2P
        
    elif opName == "smooth":
        ftAll = []
        for ft in data:        
            ft1A = ft.copy()
            for i1 in tqdm(range(ft.shape[1])):
                ft1A[:,i1] = np.convolve(ft[:,i1],np.ones(winL)/float(winL),"same")
            ftAll.append(ft1A)
        return ftAll
    
    elif opName == "adjDiff":
        ftAll = []
        for ft in data:        
            ft1D = ft[1:] - ft[:-1]
            ft1D = np.vstack([ft1D[0],ft1D])
            ftAll.append(ft1D)
        return ftAll
    
    elif opName == "winDiff":
        ftAll = []
        for ft in data:        
            ft1D = ft[winL:] - ft[:-winL]
            ft1D = np.vstack([ft1D[:winL//2],ft1D,ft1D[-winL//2:]])
            ftAll.append(ft1D)            
        return ftAll
    
    elif opName == "conv":
        v = (-1.0*np.ones(winL))/(winL/2.0)
        v[:winL//2] *= -1        

        ftAll = []
        for ft in data:
            ftC = []#ft1.copy(), ft2.copy()
            # note that np.convolve flips the v vector hence sign is inverted above
            for i1 in tqdm(range(ft.shape[1])):
                ftC.append(np.convolve(ft[:,i1],v,"same"))
            ftC = np.array(ftC).transpose()
            
            # a forced fix for zero-sum delta descs -> use the raw descriptor (could otherwise skip frames)
            ftC = np.array([ft[j] if f.sum()==0 else f for j,f in enumerate(ftC)])
            
            ftAll.append(ftC)
            
        return ftAll        
    
    return

def performOpsMulti(data,opList,winL=None,pcaComp=None):
    for opName in opList:
        data = performOps(data,opName,winL,pcaComp)
    return data