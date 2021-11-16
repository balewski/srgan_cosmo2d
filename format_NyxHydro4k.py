#!/usr/bin/env python
""" 
format  raw  Nyx hydro simulation, 4096^3
for ML training
2k cube takes: 3 min & 64 GB

For 4k cube one needs 256 GB of RAM, do it on big memory node

module load cmem
salloc -C amd -q bigmem -t 1:00:00

"""

from toolbox.Util_H5io3 import write3_data_hdf5

from pprint import pprint
import numpy as np
import h5py,time

#...!...!..................
def read_meta(inpF,verb=1):
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
def read_bigCube(inpF,name,size,verb=1):
    start=time.time()
    f = h5py.File(inpF,'r')
    if verb>1: print('f keys:',sorted(f.keys()))
    dataN="native_fields/%s"%name
    print('read %s ...'%dataN, 'size=',size)
    data = f[dataN][:size,:size,:size,]
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
    inpF="/global/cscratch1/sd/zarija/4096/plt01139.hdf5"
    #inpF="/global/homes/b/balewski/prje/data_NyxHydro4k/raw/plt01139.hdf5"

    outPath='/global/homes/b/balewski/prje/data_NyxHydro4k/B/'
                   
    meta=read_meta(inpF)
    cubesize=meta.pop('size')
    cubeshape=meta.pop('shape')
    meta['cell_size']=cubesize[0]/cubeshape[0]
    meta['cell_size_unit']='Mpc/h'

    bigD={}
    data=read_bigCube(inpF,"dm_density",1024*4)
    bigD['dm_density']=data 
    mxRho= np.max(data)
    print('cube min/max',np.min(data),mxRho)

    meta['max_rho']=float(mxRho)
    meta['cube_shape']=list(data.shape)
        
    pprint(meta)

    outF=outPath+'xxdm_density_%d.h5'%data.shape[0]
    write3_data_hdf5(bigD,outF,metaD=meta, verb=1)
