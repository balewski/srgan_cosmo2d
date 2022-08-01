#!/usr/bin/env python3
'''
 pack 6 hd5 for  z in [200,5,3] x [LR,HR] into a single h5

'''

import numpy as np
import argparse,os,time
from pprint import pprint
from toolbox.Util_H5io4 import read4_data_hdf5, write4_data_hdf5

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    
    parser.add_argument("--inpPath",default='/global/homes/b/balewski/prje/superRes-Nyx2022a/sixpack_cubes',help="sixpack HyxHydro cubes data") 
        
    args = parser.parse_args()    
    args.outPath=os.path.join(args.inpPath,'../')
    
    for arg in vars(args):  print( 'myArgs:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    return args


#...!...!..................
def slice_one_sixpack(inpF,verb):
    sixD,sixMD=read4_data_hdf5(inpF,verb=verb)
    sizeD=sixMD['cube_bin']
    binFact=sizeD['HR']//sizeD['LR']
    binOff=binFact//2
    #print('binfact', binFact,binOff)
    stackD={}
    for x in sixD:
        if 'HR' in x: # decimate HR cubes
            stack=sixD[x][binOff::binFact]            
        else:
            stack=sixD[x]
        #print('ss',x,stack.shape)
        stackD[x]=stack
    return stackD,sixMD


#...!...!..................
def prime_output_domain(domD,stackD,domN,nSix):
    for x in stackD:
         stack=stackD[x]
         nOne,ny,nz=stack.shape
         recN='%s_%s'%(domN,x)
         domD[recN]=np.zeros((nOne*nSix,ny,nz),dtype=np.float32)
         #print('new ',recN,domD[recN].shape)
    return domD
    
#...!...!..................
def scan_input():
    sixL=[]
    sPath=args.inpPath
    for sChild in os.listdir(sPath):
        assert 'h5' in sChild
        #print('new cube:',sChild)
        sixL.append(sChild)
    return sixL

#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    sixL=scan_input()
    print('M: found %d six-cubes'%len(sixL))
    domSplit={0:['train',12],12:['val',1],13:['test',1]}
    
    domD={}
    seedL=[]
    for ic,cubeN in enumerate(sixL):
        print('M: assemble ',ic,cubeN)
        inpF=os.path.join(args.inpPath,cubeN )
        stackD,sixMD=slice_one_sixpack(inpF,verb=ic==0)
        if ic==0: metaD=sixMD
        seedL.append(sixMD.pop('ic_seed'))
        if ic in domSplit:
            domN,nSix=domSplit[ic]
            ic0=ic
            prime_output_domain(domD,stackD,domN,nSix)
            print(domN,domD.keys())
        # insert cubes to big arrays
        for x in stackD:
            stack=stackD[x]
            nOne,ny,nz=stack.shape
            recN='%s_%s'%(domN,x)
            j=(ic-ic0)*nOne
            #print('rr',recN,ic,j)
            domD[recN][j:j+nOne]=stack
        #break # just one sixpack, for tesitng
    print('M: all slices collected',domD.keys())

    # finalize meta-data    
    metaD['seeds']=seedL
    nSix=len(seedL)
    pprint(metaD)
    outF=os.path.join(args.outPath,'sliced-Nyx2022a-c%d.h5'%nSix)
    write4_data_hdf5(domD,outF, metaD=metaD)
    print('M: done, nSix=',nSix)
        
