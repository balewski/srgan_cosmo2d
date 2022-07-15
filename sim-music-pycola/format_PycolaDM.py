#!/usr/bin/env python
""" 
format  raw  music+PyCola+DMdensity simu run by Jan in 2022-04
for ML training
See for the details: https://docs.google.com/document/d/1UwJzgpFdRGDVhWcpSciczOE61G2GrLZNETqBcOGK0cQ/edit?usp=sharing

module load cmem
salloc -C amd -q bigmem -t 2:00:00
module load pytorch
20c*L7
./format_PycolaDM.py -j 58451467_1/out_[0-14] 58451467_2/out_[0-13] -p univL7

30c* L9
./format_PycolaDM.py -j 58364013_4/out_[0-29]

390c* L9
./format_PycolaDM.py -j 58364013_4/out_[0-29] 58363933_1/out_[0-119]  58363933_2/out_[0-119]  58363933_3/out_[0-119]


"""

from toolbox.Util_H5io3 import read3_data_hdf5,write3_data_hdf5
from toolbox.Util_IOfunc import expand_dash_list

from pprint import pprint
import numpy as np
import time
import os

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verb",type=int,choices=[0, 1, 2],
                    help="increase output verbosity", default=1)
    parser.add_argument("-p", "--namePrefix",default="univL9",help="job name prefix")
    parser.add_argument("--rawPath",help="raw input  path",
                        default='/global/cscratch1/sd/balewski/univers2/'
                        )
    parser.add_argument("--outPath",help="output path",
                        default='/global/cscratch1/sd/balewski/data_univ_cola_dm2d/'
                        )
    parser.add_argument("-j", "--jobIds",nargs="+",default=['58364013_4/out_[0-29]','58363933_1/out_[0-51]'],
                        help=" blank separated list of job IDs, takes *[n1-n2]")
 
    args = parser.parse_args()
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args



#...!...!..................
def M_read_oneCube(jobNk,k):
    isFirst=k==0

    #...... split into  domains:  8/1/1
    if k%10==0: dom='valid'
    elif k%10==1: dom='test'
    else: dom='train'
    
    if k%10==0: print("ROJ read:",jobN,k,dom)
    preN=args.namePrefix
    coreN='%s_%s.dm.h5'%(preN,jobN.replace('/out',''))
    inpF=os.path.join(args.rawPath,'%s_%s'%(preN,jobN),coreN)
    #print('core:',coreN, 'inp:',inpF)

    blob,meta=read3_data_hdf5(inpF,verb=isFirst )
    seed9=meta['random']['seed[9]']
    assert seed9 not in seedS  # verify no duplicated seed occured
    seedS.add(seed9)

    zrsL=["z50","z0"]  # initial and final state tag 
    #zMap={"z49":"hr.ini","z0":"hr.fin"}  # meaning in SRGAN context

    #pprint(meta)
    #... check for overlflows during projection
    for zrs in zrsL:
        key='dm.'+zrs
        if isFirst: print('clip test:',key,meta[key])
        assert 'clip' not in meta[key]

    # ..................
    if isFirst:  # drop last few qubes if not M*10
        tenCnt=numRawCube//10
        trainCnt=8*tenCnt
        print('\nROJ create output storage nRaw=%d tenCnt=%d trainCnt=%d nSkip=%d'%(numRawCube,tenCnt,trainCnt,numRawCube-tenCnt*10))
        assert tenCnt>0
        args.tenCnt=tenCnt

        for xx in ['random','output']: meta.pop(xx)
        
        meta['packing']={'inp_raw_name1':inpF,'raw_cube_shape':{},'num_raw_cubes':numRawCube}        
        args.outMD=meta
        
        bigD={}               
        cube4d=blob['dm.rho4d']
        dimX,dimY,dimZ,dimR=cube4d.shape
        coreSh=(dimY,dimZ,dimR)
        nFlip=3 # I, mirror, rot90
        
        meta['packing']['raw_cube_shape']=list(cube4d.shape)
        #...... split into  domains:  8/1/1

        bigD['valid.hr']=np.zeros((tenCnt*dimX*nFlip,)+coreSh,dtype=cube4d.dtype)
        bigD['test.hr']=np.zeros_like(bigD['valid.hr'])        
        bigD['train.hr']=np.zeros((trainCnt*dimX*nFlip,)+coreSh,dtype=cube4d.dtype)
        for xx in bigD: print('created', xx,bigD[xx].shape)
        args.bigD=bigD
        args.bigIdx={} # idx will be advancing during storage
        for xx in bigD: args.bigIdx[xx]=0

    #.....  concatenate  raw cubes
    if k>= 10*args.tenCnt: return # skip last few qubes
    cube4d=blob['dm.rho4d']
    dimX=cube4d.shape[0]
    mkey=dom+'.hr'
    idx=args.bigIdx[mkey]
    #print('iii',idx,dom,jobNk)
    args.bigD[mkey][idx:idx+dimX]=cube4d
    idx+=dimX

    # add flips of cube
    args.bigD[mkey][idx:idx+dimX]=np.swapaxes(cube4d,0,1)
    idx+=dimX
    
    args.bigD[mkey][idx:idx+dimX]=np.swapaxes(cube4d,0,2)
    idx+=dimX    
    args.bigIdx[mkey]=idx
    return 

#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    args=get_parser()

    jobL=expand_dash_list(args.jobIds)
    numRawCube=len(jobL)
    print('M: expanded %d job(s) list'%numRawCube,jobL)

    seedS=set()
    args.outMD=None
    t0=time.time()
    for k,jobN in enumerate(jobL):
        M_read_oneCube(jobN,k)
    t1=time.time()
    print('M: all raw read in, seeds:',seedS)
    print('M: read elaT=%.1f (min)'%((t1-t0)/60.))
    #pprint(args.outMD)
    print('M: end bigIdx',args.bigIdx)

    args.outMD['packing']['big_index']=args.bigIdx
    args.outMD['project_dm']['author']='Jan'
    
    outN='%scola_dm2d_202204_c%d.h5'%(args.namePrefix,args.tenCnt*10)
    outF=os.path.join(args.outPath,outN)
    write3_data_hdf5(args.bigD,outF,metaD=args.outMD, verb=1)

    print('M:done')
    exit(0)

    # testing if MD are sane
    from toolbox.Util_IOfunc import  write_yaml
    #pprint(args.outMD)
    outF='xx.yaml'
    write_yaml(args.outMD,outF)
