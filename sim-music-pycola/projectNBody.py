#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# project DM vector field to density
import os,sys,time
import numpy as np
import h5py    
from pprint import pprint
from toolbox.Util_H5io3 import read3_data_hdf5, write3_data_hdf5

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--dataPath",default='out')
    parser.add_argument("--outPath", default='same')
    parser.add_argument("--dataName",  default='univ_base8', help="[.cola.h5] PyCola output")
    args = parser.parse_args()
    args.save_uint8=True

    if args.outPath=='same': args.outPath=args.dataPath
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def projectOne(bigD,boxlength,level,core):
    nbins=1<<level
   
    px = bigD[core+'.px']
    py = bigD[core+'.py']
    pz = bigD[core+'.pz']

    print ('pxyz: ',px[0][0][0], py[0][0][0], pz[0][0][0])
   
    #### Try using this hp.histogramdd function...
    ### For this I need to turn the particl elists into coord lists, 
    ### so (  (px[i][j][k], py[i][j][k], pz[i][j][k]), ....)
    pxf = np.ndarray.flatten(px)
    pyf = np.ndarray.flatten(py)
    pzf = np.ndarray.flatten(pz)

    print ('pxf.shape', pxf.shape)
    if args.verb>1:
        print ('pxf sample', pxf[0], pyf[0], pzf[0])
        print ('pxf min/max', pxf.min(), pxf.max())
        print ('pyf min/max', pyf.min(), pyf.max())
        print ('pzf min/max', pzf.min(), pzf.max())

    ### so the flattening is working. Now make this into a 3d array...
    ps = np.vstack( (pxf, pyf, pzf) ).T
    
    del(pxf); del(pyf); del(pzf)

    print ("one big vector list ", ps.shape, ps[77,:],'\naccumulate 3D histo...')
    
    ## OK! Then this is indeed a big old array. Now I want to histogram it.
    ## this step goes from a set of parcile coordinates to a histogram of particle counts 
    
    H, bins = np.histogramdd(ps, nbins, range=((0,boxlength),(0,boxlength),(0,boxlength)) )
 
    print ("histo shape", H.shape, 'example:', H[5][5][5])
    print ('mass sum=%.3g, min=%.3g max=%.3g'%(np.sum(H),H.min(),H.max())) # takes many seconds, just for QA
    
    return H.astype('float32')


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":
    args=get_parser()

    inpF=os.path.join(args.dataPath,args.dataName+'.cola.h5')
    bigD,inpMD=read3_data_hdf5(inpF, verb=1)
    #pprint(inpMD)
    zRedL=inpMD['pycola']['zRedShift_label']
    boxlength = int(inpMD['setup']['boxlength'])  # in Mpc/h
    level = int(inpMD['setup']['levelmax'])  
    
    outL=[]
    for zrs in zRedL:
        
        rho=projectOne(bigD,boxlength,level,zrs)
        a=rho.min(); b=rho.max();
        xD={'min':float(a),'max':float(b)}
        if args.save_uint8:
            if b>255:
                rho=np.clip(rho,0,255)
                xD['clip']=True
            rho=rho.astype('uint8')
            inpMD['dm.'+zrs]=xD
        outL.append(rho)
        #ok11

    rho4d=np.stack(outL, axis=-1)
    print('M:rho4d:',rho4d.shape)
    bigD={'dm.rho4d':rho4d}
    outF=os.path.join(args.outPath,args.dataName+'.dm.h5')
    
    write3_data_hdf5(bigD,outF,metaD=inpMD)
    #pprint(inpMD)
    print('M: done')
