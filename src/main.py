import numpy as np
import os, time, argparse

from scipy.spatial.distance import cdist

import ops, outFuncs

parser = argparse.ArgumentParser(description="DeltaDescriptors", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# task args
parser.add_argument("--genDesc","-gd", action="store_true", help="Use this flag to compute descriptors.")
parser.add_argument("--genMatch", "-gm", action="store_true", help="Use this flag to generate matches for a given pair of descriptors.")
parser.add_argument("--eval", "-e", action="store_true", help="Use this flag to evaluate match output.")

# path args
parser.add_argument("--descFullPath1", "-ip1", type=str, help="Path to the descriptor data file.")
parser.add_argument("--descFullPath2", "-ip2", type=str, help="Path to the 'query' descriptor data file.")
parser.add_argument("--outPath", "-op", type=str, default="./out/",help="Path to store output.")
parser.add_argument("--matchOutputPath", "-mop", type=str, help="Path to the match output.")
parser.add_argument("--cordsPath1", "-cp1", type=str, help="Path to reference image co-ordinates.")
parser.add_argument("--cordsPath2", "-cp2",type=str, help="Path to query image co-ordinates.")

# param args
parser.add_argument("--seqLength", "-l", type=int, default=16, help="Sequential span (in frames) to compute delta or smooth descriptors.")
parser.add_argument("--descOpType", "-d", type=str, default="delta", help="Descriptor type to compute.", choices = ["delta","smooth"])


def getFN(fullPath):
    return os.path.splitext(os.path.basename(fullPath))[0]

def main():
    opt = parser.parse_args()
    print(opt)
    
    if opt.genMatch and opt.descFullPath2 is None:
        raise Exception("For generating distance matrix, provide path to query descriptors as well.")

    timeStamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    outDir = os.path.join(opt.outPath,"results_{}/".format(timeStamp))
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Created output directory: ", outDir)

    if opt.genDesc or opt.genMatch:
        descData = []
        descR = np.load(opt.descFullPath1)
        descData += [descR]
        if opt.genMatch:
            descQ = np.load(opt.descFullPath2)
            descData += [descQ]

    if opt.genDesc:
        print("Computing Descriptors...")
        descData = ops.performOps(descData,opt.descOpType,opt.seqLength)
        
        # store descriptors
        np.save( os.path.join(outDir, getFN(opt.descFullPath1) + "_" + opt.descOpType ), descData[0])
        if opt.genMatch:
            np.save( os.path.join(outDir, getFN(opt.descFullPath2) + "_" + opt.descOpType ), descData[1])

    if opt.genMatch:
        print("Generating Matches...")
        dMat = cdist(descData[0],descData[1],"cosine")
        
        print("Distance Matrix Shape: ",dMat.shape)

        mInds = np.argmin(dMat,axis=0)
        mDists = dMat[mInds,np.arange(len(mInds))]
        
        # store match output 
        np.savez(os.path.join(outDir,"matchOutput.npz"),matchInds=mInds,matchDists=mDists)
        outFuncs.saveDistMat(dMat,mInds,os.path.join(outDir,"matchMat.jpg"))

    if opt.eval:
        if opt.genMatch:
            resPath = os.path.join(outDir,"matchOutput.npz")
        else:
            resPath = opt.matchOutputPath
        pAt100R = outFuncs.evaluate(resPath,opt.cordsPath1,opt.cordsPath2)

    return


if __name__ == "__main__":
    
    main()
