#!/usr/bin/env python
""" 
Inspect  raw  Nyx hydro simulation

"""

from pprint import pprint
import numpy as np
import h5py,time,os

#...!...!..................
def read_one_nyx_h5(inpF,field_name):
    meta=read_h5meta(inpF)
    cubesize=meta.pop('size')
    cubeshape=meta.pop('shape')
    meta['cell_size']=cubesize[0]/cubeshape[0]
    meta['cell_size_unit']='Mpc/h'

    bigD={}
    
    data=read_h5bigCube(inpF,field_name)
    bigD[field_name]=data 
    mxRho= np.max(data)
    print('cube min/max',np.min(data),mxRho)

    meta['max_rho']=float(mxRho)
    meta['cube_shape']=list(data.shape)
    return bigD,meta

#...!...!..................
def read_h5meta(inpF,verb=1):
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
        outD[x]=f[g].attrs[x]
    f.close()
    return outD


#...!...!..................
def read_h5bigCube(inpF,name,verb=1):
    start=time.time()
    f = h5py.File(inpF,'r')
    if verb>1: print('f keys:',sorted(f.keys()))
    dataN="native_fields/%s"%name
    print('read %s ...'%dataN, 'shape=',f[dataN].shape)
    data = f[dataN][:]
    f.close()
    
    #print('bd',data.shape,data.dtype)
    if verb>0:
        print(' done read h5,   elaT=%.1f sec'%((time.time() - start)))
    return data


#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__ == '__main__':
    dataPath='/pscratch/sd/b/balewski/tmp_NyxProd/2760607_2univ/cube_82337823'
    inpF=os.path.join(dataPath,"plotLR00001_converted.h5")  
    print('M: inpF',inpF)

    fieldN="dm_density"
    bigD,meta=read_one_nyx_h5(inpF, fieldN)
          
    pprint(meta)

    #outF=outPath+'xxdm_density_%d.h5'%data.shape[0]
    #write3_data_hdf5(bigD,outF,metaD=meta, verb=1)
    print('M: done')
