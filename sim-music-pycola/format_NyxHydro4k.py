#!/usr/bin/env python
""" 
format   NyxHydro4k data produced by Zarija

This cube has only HR density at the final state.
./format_NyxHydro4k.py --cutLevel 9 


"""

from toolbox.Util_H5io3 import write3_data_hdf5

import h5py,time
from pprint import pprint
import numpy as np
import time
import os

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int,choices=[0, 1, 2],
                    help="increase output verbosity", default=1)
    parser.add_argument("-p", "--namePrefix",default="NyxHydro4k",help="simu name prefix")
    parser.add_argument("--rawPath",help="raw input  path",
                        default='/global/homes/b/balewski/prje/data_NyxHydro4k/raw/'
                        )
    parser.add_argument("--rawName",default='plt01139',help="[.hdf5] Zaria's file")
    parser.add_argument("--cutLevel",default=7, type=int,help='2^level is the output 2D size')
    
    parser.add_argument("--outPath",help="output path",
                        default='/global/homes/b/balewski/prje/data_NyxHydro4k/B/'
                        )
    args = parser.parse_args()
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args


#...!...!..................
def read_cubeMeta(inpF,verb=1):
    start=time.time()
    f = h5py.File(inpF,'r')
    print('f keys:',sorted(f.keys()))

    outD={}
    g='domain'
    if verb>1: print('g=',g,f[g])
    atrL=f[g].attrs.keys()
    if verb>1: print('atr1',atrL)

    for x in atrL:
        print(x,f[g].attrs[x])
        outD[x]=f[g].attrs[x][:].tolist()
        
    g='universe'
    atrL=f[g].attrs.keys()
    if verb>1: print('atr2',atrL)
    for x in atrL:
        print(x,f[g].attrs[x])
        outD[x]=float(f[g].attrs[x])
    f.close()
    return outD

#...!...!..................
def read_cubeData(inpF,F,size,verb=1):
    iRed=1
    start=time.time()
    f = h5py.File(inpF,'r')
    if verb>1: print('f keys:',sorted(f.keys()))
    dataN="native_fields/dm_density"
    print('read %s ...'%dataN, 'size=',size)
    ix=0
    mx=F.hr_valid.shape[0]
    print('read val...',ix,mx)
    F.hr_valid[...,iRed]=f[dataN][:mx,:size,ix:ix+size]
    ix+=size
    print('read test...',ix,mx)
    F.hr_test[...,iRed]=f[dataN][:mx,:size,ix:ix+size]
    ix+=size
    mx=F.hr_train.shape[0]
    print('read train...',ix,mx)
    F.hr_train[...,iRed]=f[dataN][:mx,:size,ix:ix+size]
    
    f.close()
    
    #print('bd',data.shape,data.dtype)
    if verb>0:
        print(' done read h5,   elaT=%.1f sec'%((time.time() - start)))


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    inpF=os.path.join(args.rawPath,args.rawName+'.hdf5')
    metaG=read_cubeMeta(inpF,args.verb)
    cubesize=metaG.pop('size')
    cubeshape=metaG.pop('shape')
    metaG['box_size']=cubesize[0]
    metaG['cell_size']=cubesize[0]/cubeshape[0]
    metaG['size_unit']='Mpc/h'
    metaG['author']='Zaria'

    
    hr_size=1<<args.cutLevel
    numRed=2 # only index=1=Fin is used, 0=Ini is left empty
    num_small_cubes=cubeshape[0]//hr_size
    num_samp=num_small_cubes*hr_size
    print('ww',hr_size,num_samp)
    metaP={}
    metaP={'inp_raw_name1':inpF,'raw_cube_shape':list(cubeshape),'num_small_cubes':num_small_cubes,'cutLevel':args.cutLevel}
    pprint(metaP)

    
    # clever list-->numpy conversion, Thorsten's idea
    class Empty: pass
    F=Empty()  # fields (not images)
    F.hr_train=np.zeros([num_samp,hr_size,hr_size,numRed],dtype=np.float32)
    F.hr_valid=np.zeros([num_samp//8,hr_size,hr_size,numRed],dtype=np.float32)
    F.hr_test=np.empty_like(F.hr_valid)
    print('F-container, train',F.hr_train.shape,'val:',F.hr_valid.shape,list(F.__dict__))
    read_cubeData(inpF,F,hr_size,args.verb)

    bigD=vars(F)
    metaS={'hr_size':hr_size}
    meta3D={'boxlength':metaG['box_size']}
    outMD={'packing':metaP,'sim3d':metaG,'num_inp_chan':1,'data_shape':metaS,'setup':meta3D}
    outN='%s_L%d_dm2d_202106.h5'%(args.namePrefix,args.cutLevel)
    outF=os.path.join(args.outPath,outN)
    write3_data_hdf5(bigD,outF,metaD=outMD, verb=1)

    print('M:done')
    pprint(outMD)
    
    #exit(0)

    # testing if MD are sane
    from toolbox.Util_IOfunc import  write_yaml
    outF='xx.yaml'
    write_yaml(outMD,outF)
