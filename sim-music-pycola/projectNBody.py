#!/usr/bin/env python3
from __future__ import print_function
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

import numpy as np
import math
import sys
from pycola3_OmSiNs import read_yaml
import os

### This code will take the output of one NBody simulation (ie from the pycola code), and split it into ? sub-volumes, and histogram them. 


######## Loop over the files!
def projectOne(infile,boxlength,levelmax,coreName):
    nbins=1<<levelmax
    histFile=coreName+'_dim%d_full'%nbins
    print('input=',infile,'boxlength=',boxlength,' nbins=',nbins,'histFile=',histFile)
    ### First, read in the px/py/pz from the pycola output file
    data = np.load(infile)

    px = data['px']
    py = data['py']
    pz = data['pz']

    print ('pxyz one: ',px[0][0][0], py[0][0][0], pz[0][0][0])
   

    #### Try using this hp.histogramdd function...
    ### For this I need to turn the particl elists into coord lists, 
    ### so (  (px[i][j][k], py[i][j][k], pz[i][j][k]), ....)
    pxf = np.ndarray.flatten(px)
    pyf = np.ndarray.flatten(py)
    pzf = np.ndarray.flatten(pz)

    print ('pxf.shape', pxf.shape)
    print ('pxf sample', pxf[0], pyf[0], pzf[0])
    print ('pxf min/max', pxf.min(), pxf.max())

    ### so the flattening is working. Now make this into a 3d array...
    ps = np.vstack( (pxf, pyf, pzf) ).T
    
    del(pxf); del(pyf); del(pzf)

    print ("one big vector list ", ps.shape, ps[77,:],'\naccumulate 3D histo...')
    
    ## OK! Then this is indeed a big old array. Now I want to histogram it.
    ## this step goes from a set of parcile coordinates to a histogram of particle counts 
    
    H, bins = np.histogramdd(ps, nbins, range=((0,boxlength),(0,boxlength),(0,boxlength)) )
 
    print ("histo dshape!", H.shape,  H[0][0][0])
    print ('mass sum=%.3g'%np.sum(H)) # takes many seconds, just for QA
    np.save(histFile, H)
    return H
# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 
# - - - - - - - - - - - - - - - - - - - - - - 

from pprint import pprint
if __name__ == '__main__':


    ioPath=sys.argv[1]
    ymlF=sys.argv[2]
    print ("read YAML from ",ymlF,' and pprint it:')
    
    blob=read_yaml(ymlF)

    pprint(blob)
    core=blob['coreStr']
    vectFile=os.path.join(ioPath,blob['coreStr']+'.npz')
    fnameOut=vectFile.replace('.npz','')

    bigH=projectOne(vectFile, blob['boxlength'],blob['levelmax'],fnameOut)
    print('projection completed:',fnameOut)
    
