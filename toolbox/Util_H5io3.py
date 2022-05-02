#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# = = = = =  HD5 advanced storage = = =
#  can hold arbitrary numpy array
#  can hold python dictionaries
#  can write single float or int w/o np-array packing

import numpy as np
import h5py, time, os
import json
from pprint import pprint

#...!...!..................
def write3_data_hdf5(dataD,outF,metaD=None,verb=1):
    if metaD!=None:
        if verb>1:
            print('WH5: metaD dump, keys:', sorted( metaD))
            for x in metaD:
                val=metaD[x]
                print('k=',x,type(val),', val=',val)
                json.dumps(val)  # try JSON-ify
        metaJ=json.dumps(metaD)
        #print('meta.JSON:',metaJ)
        dataD['meta.JSON']=metaJ
    
    dtvs = h5py.special_dtype(vlen=str)
    h5f = h5py.File(outF, 'w')
    if verb>0:
            print('saving data as hdf5:',outF)
            start = time.time()
    for item in dataD:
        rec=dataD[item]
        if verb>1: print('x=',item,type(rec))
        if type(rec)==str: # special case
            dset = h5f.create_dataset(item, (1,), dtype=dtvs)
            dset[0]=rec
            if verb>0:print('h5-write :',item, 'as string',dset.shape,dset.dtype)
            continue
        if type(rec)!=np.ndarray: # packs a single value in ot np-array
            rec=np.array([rec])
        h5f.create_dataset(item, data=rec)
        if verb>0:print('h5-write :',item, rec.shape,rec.dtype)
    h5f.close()
    if verb<1: return
    xx=os.path.getsize(outF)/1048576
    print('closed  hdf5:',outF,' size=%.2f MB, elaT=%.1f sec'%(xx,(time.time() - start)))
    
#...!...!..................
def read3_data_hdf5(inpF,verb=1,skipKey=None):
    if verb>0:
            print('read hdf5 from :',inpF)
            if skipKey!=None:  print('   h5 skipKey:',skipKey)
            start = time.time()
    h5f = h5py.File(inpF, 'r')
    objD={}
    for x in h5f.keys():
        if verb>1: print('item=',x,type(h5f[x]),h5f[x].shape,h5f[x].dtype)
        if skipKey!=None:
            skip=False            
            for y in skipKey:
                if y in x: skip=True
            if skip: continue
        if h5f[x].dtype==object:
            obj=h5f[x][0]
            #print('bbb',type(obj),obj.dtype)
            if verb>0: print('read str:',x,len(obj),type(obj))
        else:
            obj=h5f[x][:]
            if verb>0: print('read obj:',x,obj.shape,obj.dtype)
        objD[x]=obj
    try:
        inpMD=json.loads(objD.pop('meta.JSON'))
        if verb>0: print('recovered meta-data with %d keys'%len(inpMD))
    except:
        inpMD=None
    if verb>0:
        print(' done h5, num rec:%d  elaT=%.1f sec'%(len(objD),(time.time() - start)))

    h5f.close()

    return objD,inpMD



#=================================
#=================================
#   U N I T   T E S T
#=================================
#=================================

if __name__=="__main__":
    from pprint import pprint
    import json

    print('testing h5IO ver 3')
    outF='abcTest.h5'
    
    var1=float(15)
    one=np.zeros(shape=5); one[3]=3
    two=np.zeros(shape=(2,3)); two[1,2]=4
    txt='This is text1'
    metaD={"age":17,"dom":"white","dates":[11,22,33]}
   
    outD={'one':one,'two':two,'var1':var1,'text':txt}

    write3_data_hdf5(outD,outF,metaD=metaD)

    # .... testing reading of H5
    
    print('\n *****  verify by reading it back from',outF)
    blob,meta2=read3_data_hdf5(outF)
    print(' recovered meta-data'); pprint(meta2)
    print('dump read-in data')
    for x in blob:
        print('\nkey=',x); pprint(blob[x])


    print('\n check raw content:   h5dump %s\n'%outF)
