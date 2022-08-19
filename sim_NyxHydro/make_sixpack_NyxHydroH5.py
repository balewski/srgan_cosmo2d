#!/usr/bin/env python3
'''
 pack 6 hd5 for  z in [200,5,3] x [LR,HR] into a single h5

'''

import scipy.stats as stats
from inspect_NyxHydroH5 import read_one_nyx_h5
import numpy as np
import argparse,os,time
from pprint import pprint
from toolbox.Util_H5io4 import write4_data_hdf5

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    
    parser.add_argument("--basePath",default='/pscratch/sd/b/balewski/tmp_NyxProd',help="head dir for HyxHydro data") 
    parser.add_argument("--jobName",default='2760607_2univ',help="head dir for HyxHydro data")
    
    args = parser.parse_args()
    
    args.inpPath=os.path.join(args.basePath,args.jobName)
    args.outPath=os.path.join(args.basePath,'sixpack')
    
    for arg in vars(args):  print( 'myArgs:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args

#...!...!..................
def scan_input():
    genD={}
    sPath=args.inpPath
    for sChild in os.listdir(sPath):                
        sChildPath = os.path.join(sPath,sChild)
        if not os.path.isdir(sChildPath): continue
        print('new cube:')
        cubeN=sChild
        genD[cubeN]={'HR':[],'LR':[], 'path': sChildPath}
        for gChild in sorted(os.listdir(sChildPath)):
            if 'h5' not in gChild: continue
            print(sChild,gChild)
            for hlr in hlL:
                if hlr not in gChild: continue
                genD[cubeN][hlr].append(gChild)
    return genD
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    hlL=['HR','LR']
    fieldL= ['baryon_density','dm_density','particle_vx','particle_vy','particle_vz',
             'temperature','velocity_x','velocity_y','velocity_z']
    
    genD=scan_input()
    pprint(genD)

    nSix=0
    for cubeN in genD:
        verb=1
        print('M: assemble ',cubeN)
        ipath=genD[cubeN]['path']
        bigD={}
        for hlr in hlL:
            genD[cubeN][hlr+'m']=[]
            for x in genD[cubeN][hlr]:
                inpF=os.path.join( ipath,x)
                big,meta=read_one_nyx_h5(inpF,fieldL,verb); verb=0
                genD[cubeN][hlr+'m'].append(meta)
                #pprint(meta)
                zRed=meta['redshift']
                #print(big.keys())
                #print('zRed=%.f'%zRed)
                for x in big.keys():
                    name='%s_%s_z%.0f'%(x,hlr,zRed)
                    print('M: add',name)
                    bigD[name]=big[x]

        # construct common  meta-data
            
        cs={ hlr:genD[cubeN][hlr+'m'][0]['cell_size']  for hlr in hlL }
        bs={ hlr:genD[cubeN][hlr+'m'][0]['cube_shape'][0]  for hlr in hlL }
        rs=[ int('%.0f'%x['redshift']) for x in  genD[cubeN][hlr+'m']]
        
        seed=cubeN.split('_')[1]
        #print(rs,seed)
        #pprint(cs)
        #pprint(bs)
        
        meta.pop('cube_shape')
        metaD=meta
        metaD['redshift']=rs
        metaD['cell_size']=cs
        metaD['cube_bin']=bs
        metaD['ic_seed']=seed
        pprint(metaD)
        outF=os.path.join(args.outPath,cubeN+'.sixpack.h5')
        write4_data_hdf5(bigD,outF, metaD=metaD)
        nSix+=1
    print('M: done, nSix=',nSix)
        
