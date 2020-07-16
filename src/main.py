import numpy as np
import os, time, argparse

from scipy.spatial.distance import cdist

import ops, outFuncs

parser = argparse.ArgumentParser(description="DeltaDescriptors", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--genDesc", action="store_true", help="Use this flag to compute descriptors.")
parser.add_argument("--descFullPath", type=str, help="Path to the descriptor data file.")
parser.add_argument("--seqLength", type=int, default=64, help="Sequential span (in frames) to compute Delta Descriptors.")
parser.add_argument("--outPath", type=str, default="./out/",help="Path to store output.")
parser.add_argument("--descOpType", type=str, default="Delta", help="Descriptor type to compute.", choices = ["Delta","Smooth"])

parser.add_argument("--genMatch", action="store_true", help="Use this flag to generate matches for a given pair of descriptors.")
parser.add_argument("--descQueryFullPath", type=str, help="Path to the 'query' descriptor data file.")

parser.add_argument("--eval", action="store_true", help="Use this flag to evaluate match output.")
parser.add_argument("--matchOutputPath", type=str, help="Path to the match output.")
parser.add_argument("--cordsPathR", type=str, help="Path to reference image co-ordinates.")
parser.add_argument("--cordsPathQ", type=str, help="Path to query image co-ordinates.")

def main():
    opt = parser.parse_args()
    print(opt)
    
    if opt.genMatch and opt.descQueryFullPath is None:
        raise Exception("For generating confusion (distance) matrix, provide path to query descriptors as well.")

    timeStamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    outDir = os.path.join(opt.outPath,"results_{}/".format(timeStamp))
    if not os.path.exists(outDir):
        os.makedirs(outDir)
        print("Created output directory: ", outDir)

    if opt.genDesc or opt.genMatch:
        descData = []
        descR = np.load(opt.descFullPath)
        descData += [descR]
        if opt.genMatch:
            descQ = np.load(opt.descQueryFullPath)
            descData += [descQ]

    if opt.genDesc:
        print("Computing Descriptors...")
        if opt.descOpType == "Delta":
            opName = "conv"
        elif opt.descOpType == "Smooth":
            opName = "smooth"

        descData = ops.performOps(descData,opName,opt.seqLength)
        
        # store descriptors
        np.save( os.path.join(outDir, os.path.split(opt.descFullPath)[-1] + "_" + opt.descOpType ), descData[0])
        if opt.genMatch:
            np.save( os.path.join(outDir, os.path.split(opt.descQueryFullPath)[-1] + "_" + opt.descOpType ), descData[1])

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
        pAt100R = outFuncs.evaluate(resPath,opt.cordsPathR,opt.cordsPathQ)

    return


if __name__ == "__main__":
    
    main()
