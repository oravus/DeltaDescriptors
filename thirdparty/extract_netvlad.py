import numpy as np
import tensorflow as tf
import cv2, sys, os, argparse
from datetime import datetime
from tqdm import tqdm


sys.path.append("./netvlad_tf_open/python/")

import netvlad_tf.nets as nets

def argParser():
    parser = argparse.ArgumentParser(description="NetVLAD Feature Extractor")

    parser.add_argument("--imPath", "-i", help="Path to image directory",type=str)
    parser.add_argument("--outPath", "-o", default="../out/", help="Full path to store output descriptor npy",type=str)
    parser.add_argument("--batchSize", "-b", default=8, help="Batch Size",type=int)

    if len(sys.argv) < 2:
        print(parser.format_help())
        sys.exit(1)    

    args=parser.parse_args()
    
    return args

def getSessVars():
    tf.reset_default_graph()

    image_batch = tf.placeholder(
            dtype=tf.float32, shape=[None, None, None, 3])

    net_out = nets.vgg16NetvladPca(image_batch)
    saver = tf.train.Saver()

    sess = tf.Session()
    saver.restore(sess, nets.defaultCheckpoint())

    return sess, net_out, image_batch


def extractFeat(sessVars,batchSize,imPath,imList,verbose=True):

    sess, net_out, image_batch = sessVars

    ims, fOuts = [], []
    for i1,name in tqdm(enumerate(imList)):
        im = cv2.imread(os.path.join(imPath,name))[:,:,::-1]
        ims.append(im)
        
        if len(ims) == batchSize or i1 == len(imList)-1:
            fOut = sess.run(net_out,feed_dict={image_batch:np.array(ims)})
            ims = []
            fOuts.append(fOut)
            
            if verbose:
                print("Processed ", i1)

    fOuts = np.concatenate(fOuts)
    if verbose:
        print("Descriptor matrix shape: ",fOuts.shape)
    return fOuts

def process(imPath,batchSize=8,outPath=None):

    imList = np.sort(os.listdir(imPath))
    sessVars = getSessVars()
    
    nvFt = extractFeat(sessVars,batchSize,imPath,imList)

    if outPath is not None:        
        if not os.path.exists(outPath):
            os.makedirs(outPath)
        uniqueTS = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        saveFullName = os.path.join(outPath,"{}_nvFt".format(uniqueTS))
        np.save(saveFullName,nvFt)
        print("Stored at: ", saveFullName)
    
    return nvFt
    
    
if __name__== "__main__":
    args = argParser()    
    
    outFt = process(args.imPath,args.batchSize,args.outPath)

