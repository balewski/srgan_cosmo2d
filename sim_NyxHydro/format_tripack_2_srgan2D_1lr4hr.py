#!/usr/bin/env python3
'''
 format  tri-pack.hd5  into a single h5
stores 4 HR slices for each LR slice
'''

import numpy as np
import argparse,os,time
from pprint import pprint
from toolbox.Util_H5io4 import read4_data_hdf5, write4_data_hdf5

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')
    
    parser.add_argument("--inpPath",
                        default='/global/homes/b/balewski/prje/superRes-Nyx2022a-flux/tripack_cubes'
                        #default='/pscratch/sd/b/balewski/tmp_NyxProd/tripack_cubes'
                        ,help="tri-pack HyxHydro cubes data") 
        
    args = parser.parse_args()    
    args.outPath=os.path.join(args.inpPath,'/tmp')
    args.outPath=os.path.join(args.inpPath,'../')
    
    for arg in vars(args):  print( 'myArgs:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    return args

#...!...!..................
def slice_one_tripack(sixD,sixMD,ir):    
    stackD={}
    assert ir in [0,1]
    #print('ii',sorted(sixD))
    for x in sixD:
        cube=sixD[x]  # ir==0  order [x,y,z] , slices: y-z
        if ir==1: #  order [y,x,z] , slices: x-z
            cube=np.swapaxes(cube,0,1)
        if 'HR' in x: #  multi-slices from HR cubes
            stack=cube.reshape(sizeD['HR']//binFact,binFact,sizeD['HR'],sizeD['HR'])
            #print('ss',stack.shape)
        else:
            stack=cube
        #print('ss',ir,x,cube.shape); print(cube[0,10,:3]); bbb
        stackD[x]=stack
    return stackD



#...!...!..................
def prime_output_domain(domD,stackD,domN,nSixRot):
    for x in stackD:
        stack=stackD[x]
        nx,ny,nz=stack.shape
        recN='%s_%s'%(domN,x)
        shp=(nx*nSixRot,ny,nz)
        if 'HR' in x: shp=(nx*nSixRot//binFact,binFact,ny,nz)
        
        domD[recN]=np.zeros(shp,dtype=np.float32)
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
    triL=scan_input_path()
    #triL=triL[:99]

    nInp=len(triL)
    print('M: found %d six-cubes at:'%nInp, args.inpPath)

    if nInp==14:
        domSplit={0:['train',12],12:['valid',1],13:['test',1]} # input 14c
    elif nInp==99:
        domSplit={0:['train',81],81:['valid',9],90:['test',9]} # input 99c
    elif nInp==213:
        domSplit={0:['train',180],180:['valid',20],200:['test',13]} # input 213c
    elif nInp==2:
        domSplit={0:['train',2],2:['valid',0],2:['test',2]} # input 2c
    else:
        fix_me44
        #domSplit={0:['train',2],2:['valid',1],3:['test',1]} # input 4c: superRes-Nyx2022a
    nRot=2 # number of rotations of the original cubes : use 1 or 2

    # reference cube name defining number of output slices for all cubes
 
    domD={}
    seedL=[]
    for ic,cubeN in enumerate(triL):
        print('M: assemble ',ic,cubeN,'nRot=',nRot)
        inpF=os.path.join(args.inpPath,cubeN )
        triD,triMD=read4_data_hdf5(inpF,verb=ic==0)
        sizeD=triMD['cube_bin']
        binFact=sizeD['HR']//sizeD['LR']
        #pprint(sixMD); ccc
                   
        if ic==0: metaD=triMD
        seedL.append(triMD.pop('ic_seed'))
        
        if ic in domSplit:  # domains handling
            domN,nTri=domSplit[ic]
            prime_output_domain(domD,triD,domN,nTri*nRot)
            ic0=ic # reset address for writeing          
            print('M:prim',domN,nTri,domD.keys())

        for ir in range(nRot):
            stackD=slice_one_tripack(triD,triMD,ir)    
            # insert cubes to big arrays
            #print('iRot=',ir,'fields:',list(triD))
            
            for x in stackD:
                stack=stackD[x]
                nOne=stack.shape[0]
                recN='%s_%s'%(domN,x)
                j=((ic-ic0)*nRot+ir)*nOne
                #print('rr',recN,ic,j)
                domD[recN][j:j+nOne]=stack
              
            #break # just one sixpack, for tesitng
    print('M: all slices collected',domD.keys())

    # finalize meta-data    
    metaD['seeds']=seedL
    metaD['upscale_factor']=binFact
    if 1: # sanity check
        for x in domD:
            tower=domD[x]
            vmin,vmax=np.min(tower),np.max(tower)
            print('tower min/max',vmin,vmax,x)
        
    pprint(metaD)
    nTri=len(seedL)
    outF=os.path.join(args.outPath,'flux-1LR%dHR-Nyx2022a-r%dc%d.h5'%(binFact,nRot,nTri))
    write4_data_hdf5(domD,outF, metaD=metaD)
    print('M: done, nTre=',nTri)
    #print('xxx',type(binFact))
        
