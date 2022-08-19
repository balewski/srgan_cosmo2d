#!/usr/bin/env python3
'''
 pack 6 hd5 for  z in [200,5,3] x [LR,HR] into a single h5

salloc  -q interactive  -t4:00:00 -A m3363 -C cpu    -N 1 

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
    args.outPath=os.path.join(args.inpPath,'/tmp')
    #args.outPath=os.path.join(args.inpPath,'../')
    
    for arg in vars(args):  print( 'myArgs:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    return args

#...!...!..................
def test_sum(recN,vol):
    sumV=np.sum(vol,axis=(1,2))
    sa=np.mean(sumV); sd=np.std(sumV)
    print('TS:',recN,vol.shape, '2d:',vol.shape[1]**2,'sum=%.1f +/- %.1f'%(sa,sd))
    #ok00
#...!...!..................
def slice_one_sixpack(sixD,sixMD,ir):    
    sizeD=sixMD['cube_bin']
    binFact=sizeD['HR']//sizeD['LR']
    binOff=binFact//2  # hardcoded 1/2 of HR/LR step
    #print('binfact', binFact,binOff)
    stackD={}
    for x in sixD:
        cube=sixD[x]
        if ir>0: cube=np.swapaxes(cube,0,ir)
        if 'HR' in x: # decimate HR cubes
            stack=cube[binOff::binFact]            
        else:
            stack=cube
        #print('ss',ir,x,cube.shape); print(cube[0,10,:3]); bbb
        stackD[x]=stack
    return stackD


#...!...!..................
def prime_output_domain(domD,stackD,domN,nSixRot,oneN):
    nOne=stackD[oneN].shape[0]
    print('prime mOne=%d'%nOne)
    for x in stackD:
        stack=stackD[x]
        _,ny,nz=stack.shape
        recN='%s_%s'%(domN,x)
        domD[recN]=np.zeros((nOne*nSixRot,ny,nz),dtype=np.float32)
        print('new out',recN,domD[recN].shape)        
    return domD


#...!...!..................
def scan_input_path():
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
    sixL=scan_input_path()
    sixL=sixL[:1]
    print('M: found %d six-cubes at:'%len(sixL), args.inpPath)
    domSplit={0:['train',12],12:['valid',1],13:['test',1]} # input 14c: superRes-Nyx2022a
    nRot=3 # number of rotations of the original cubes : use 1 or 3

    fieldN="baryon_density"
    fieldN="dm_density"

    # reference cube name defining number of output slizes for all cubes
    oneN=fieldN+'_LR_z3'

    domD={}
    seedL=[]
    for ic,cubeN in enumerate(sixL):
        print('M: assemble ',ic,cubeN,'nRot=',nRot)
        inpF=os.path.join(args.inpPath,cubeN )
        sixD,sixMD=read4_data_hdf5(inpF,verb=ic==0,acceptFilter=[fieldN])
        #pprint(sixMD); ccc
        
        if ic==0: metaD=sixMD
        seedL.append(sixMD.pop('ic_seed'))
        
        if ic in domSplit:  # domains handling
            domN,nSix=domSplit[ic]
            prime_output_domain(domD,sixD,domN,nSix*nRot,oneN)
            ic0=ic # reset address for writeing          
            print('M:prim',domN,nSix,domD.keys())
        
        for ir in range(nRot):
            stackD=slice_one_sixpack(sixD,sixMD,ir)    
            # insert cubes to big arrays
            print('iRot=',ir)
            for x in stackD:
                stack=stackD[x]
                nOne,ny,nz=stack.shape
                recN='%s_%s'%(domN,x)
                j=((ic-ic0)*nRot+ir)*nOne
                #print('rr',recN,ic,j)
                domD[recN][j:j+nOne]=stack
                test_sum(recN,stack)
            #break # just one sixpack, for tesitng
    print('M: all slices collected',domD.keys())

    # finalize meta-data    
    metaD['seeds']=seedL
    if 1: # cleanup post-factum
        metaD['size_unit']=metaD.pop('cell_size_unit')
        metaD['cube_size']=10.        
        
    pprint(metaD)
    nSix=len(seedL)
    outF=os.path.join(args.outPath,'%s-Nyx2022a-r%dc%d.h5'%(fieldN,nRot,nSix))
    write4_data_hdf5(domD,outF, metaD=metaD)
    print('M: done, nSix=',nSix)
        
