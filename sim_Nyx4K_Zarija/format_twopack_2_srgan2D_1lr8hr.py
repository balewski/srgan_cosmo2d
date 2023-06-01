#!/usr/bin/env python3
'''
 format  tri-pack.hd5  into a single h5
stores 4 HR slices for each LR slice
'''

import numpy as np
import argparse,os,time
from pprint import pprint
from toolbox.Util_H5io4 import read4_data_hdf5, write4_data_hdf5
import h5py, time, os
import json,time

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')
    
    parser.add_argument("--inpPath",
                        default='/pscratch/sd/b/balewski/tmp_Nyx2023-flux/twopack_4Kcubes'
                        ,help="two-pack 4K HyxHydro cubes data") 
        
    args = parser.parse_args()    
    args.outPath='/tmp'
    args.outPath='/pscratch/sd/b/balewski/tmp_Nyx2023-flux/data'
    #args.outPath=os.path.join(args.inpPath,'../')
    
    for arg in vars(args):  print( 'myArgs:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    assert os.path.exists(args.outPath)
    return args


#...!...!..................
def slice_one_tripack(sixD,ir):    
    stackD={}
    assert ir in [0,1]
    #print('ii',sorted(sixD))
    for x in sixD:
        cube=sixD[x]  # ir==0  order [x,y,z] , slices: y-z
        if ir==1: #  order [y,x,z] , slices: x-z
            cube=np.swapaxes(cube,0,1)
        if 'HR' in x: #  multi-slices from HR cubes
            #print('ss0',cube.shape,hrLen,binFact)
            stack=cube.reshape(lrLen,binFact,hrLen,hrLen)
            #print('ss',stack.shape)
        else:
            stack=cube
        #print('ss',ir,x,cube.shape); print(cube[0,10,:3]); bbb
        stackD[x]=stack
    return stackD



#...!...!..................
def prime_output_domain(confD,domN,nMul):    
    domD={}
    for x in confD:
        ny=confD[x]
        recN='%s_%s'%(domN,x)
        shp=(ny*nMul,ny,ny)
        if 'HR' in x: shp=(ny*nMul//binFact,binFact,ny,ny)
        #print('www',domN,nMul,shp)
        
        domD[recN]=np.zeros(shp,dtype=np.float32)
        print('***new out:',recN,domD[recN].shape)
    return domD


#...!...!..................
def inspect_inpH5(h5f):
    for x in h5f.keys():
        print('item=',x,type(h5f[x]),h5f[x].shape,h5f[x].dtype)   
        
    x='meta.JSON'  #... get meta data
    obj=h5f[x][:]
    inpMD=json.loads(obj[0])
    print('M:meta',inpMD)
    return inpMD

#...!...!..................
def open_outH5(cubeN):
    outF=os.path.join(args.outPath,'%s-1LR%dHR-Nyx4k-r%d.h5'%(cubeN,binFact,nRot))
    print('Out H5:', outF)
    h5f = h5py.File(outF, 'a')
    
    return h5f

#=================================
#=================================
#  M A I N 
#=================================
#=================================
#  salloc  -q interactive  -t4:00:00  -C cpu 
if __name__ == "__main__":
    args=get_parser()
    triL=['flux_L80_s1','flux_L80_s2'] #[:1]  # tmp: only 1 big cube 
    print('M:triL:',triL)
    nInp=len(triL)
    print('M: found %d six-cubes at:'%nInp, args.inpPath)
    #assert nInp==2  # all is hardcoded to not exceed 
    nRot=2 # number of rotations of the original cubes : use 1 or 2

    #domSplit={0:['valid',2],2:['train',4],6:['test',3]}; nSub=9  # no. sub-cubes
    domSplit={0:['valid',6],6:['train',52],58:['test',6]}; nSub=64  # no. sub-cubes  - must write to pscratch because files are 500GB
    
    nChop=4 # chop each axis
     
    for jF,cubeN in enumerate(triL):
        print('\n\n======M======: assemble ',jF,cubeN,'nRot=',nRot)
        inpF=os.path.join(args.inpPath,cubeN+'.twopack.h5' )
        h5fi = h5py.File(inpF, 'r') #.....
        inpMD=inspect_inpH5(h5fi)
        
        sizeD=inpMD['cube_bin']
        binFact=sizeD['HR']//sizeD['LR']
        
        hrCube=h5fi['flux_HR_z3']
        hrLen=hrCube.shape[0]//nChop
        lrCube=h5fi['flux_LR_z3']
        
        lrLen=lrCube.shape[0]//nChop
        h5fi.close() # .....
        
        print('sub-cube dims HR=%d  LR=%d, binFact=%d'%(hrLen,lrLen,binFact))
        confD={}
        confD['flux_HR_z3']=hrLen
        confD['flux_LR_z3']=lrLen
        partD={}
        ic0=0
        for j in range(nSub) :  # read sub-qubes, one a a time
            print('\n new input sub-cube')
            i2=j//16
            x1=j-16*i2
            i1=x1//4
            i0=x1%4            
            
            if j in domSplit:  # domains handling
                domN,nTri=domSplit[j]
                icEnd=j+nTri-1
                domD=prime_output_domain(confD,domN,nTri*nRot)
                print('M:prim',j,domN,nTri,domD.keys(),icEnd)
                ic0=j # reset relative output address for  new domain
                partD[domN]=[]
            partD[domN].append(j)
            print('pack j:',j,'i012:',i0,i1,i2,'ic0=',ic0,domN)
            twoD={}  # read data

            h5fi = h5py.File(inpF, 'r') # ......
            for hl in ['flux_LR_z3','flux_HR_z3']:
                boxLen=confD[hl]
                n0=i0*boxLen
                n1=i1*boxLen
                n2=i2*boxLen
                print(hl ,'pack n012:',n0,n1,n2)
                twoD[hl]=h5fi[hl][n0:n0+boxLen,n1:n1+boxLen,n2:n2+boxLen]
            h5fi.close() # .....
            
            for ir in range(nRot):  # cube to 2D slices
                stackD=slice_one_tripack(twoD,ir)    
                # insert cubes to big arrays
                print('iRot=',ir,'fields:',list(twoD))

                for x in stackD:
                    stack=stackD[x]
                    nOne=stack.shape[0]
                    recN='%s_%s'%(domN,x)
                    j0=((j-ic0)*nRot+ir)*nOne
                    print('read',recN,j,ic0,'j0=',j0)
                    domD[recN][j0:j0+nOne]=stack
            if j==icEnd: # save and destroy data
                h5fo=open_outH5(cubeN)
                for item in domD:
                    tower=domD[item]
                    if 0: # sanity check                        
                        vmin,vmax=np.min(tower),np.max(tower)
                        print(item, 'tower min/max',vmin,vmax,x)
                    h5fo.create_dataset(item, data=tower)
                    print('saved item',item)
                h5fo.close()
                for item in domD: domD[item]=None  # purge data from RAM
        # finalize meta-data    
        inpMD['dom_part']=partD
        inpMD['inp_4k_cube']=cubeN
        inpMD['upscale_factor']=binFact
        h5fo=open_outH5(cubeN)
        dtvs = h5py.special_dtype(vlen=str)
        metaJ=json.dumps(inpMD)
        print('meta.JSON:',metaJ)
        dset = h5fo.create_dataset('meta.JSON', (1,), dtype=dtvs)
        dset[0]=metaJ
        h5fo.close()
      
    print('M: done')

        
