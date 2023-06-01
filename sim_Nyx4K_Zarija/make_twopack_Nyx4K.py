#!/usr/bin/env python3
'''
 pack 2  cubes from Zarija in hd5  into a single h5

??z=3 I'll use [LR,HR]   record:   derived_fields/tau_red  
?? HR z=50 density filed:   native_fields/baryon_density  

'''

import scipy.stats as stats
from inspect_Nyx4K_Zarija import read_one_nyx_h5
import numpy as np
import argparse,os,time
from pprint import pprint
from toolbox.Util_H5io4 import write4_data_hdf5

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")

    parser.add_argument("--basePath",default=' /pscratch/sd/b/balewski/tmp_Nyx2023-flux/',help="head dir forformatted 4K-cube  data")
    parser.add_argument("--inpPath",default='/pscratch/sd/z/zarija/MLHydro/',help="head dir for Zarija  data")
    
    parser.add_argument("--jobName",default='L80_N4096_z3_s1',help="head dir for HyxHydro data")

    args = parser.parse_args()    
    args.outPath=os.path.join(args.basePath,'twopack_4Kcubes')    
    
    for arg in vars(args):  print( 'myArgs:',arg, getattr(args, arg))
    assert os.path.exists(args.inpPath)
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args

#...!...!..................
def Xscan_input():
    genD={}
    sPath=args.inpPath
    for sChild in os.listdir(sPath):                
        sChildPath = os.path.join(sPath,sChild)
        if not os.path.isdir(sChildPath): continue
        cubeN=sChild
        print('new cube:',cubeN)
        tmpD={'HR':[],'LR':[]}
        for gChild in sorted(os.listdir(sChildPath)):
            if 'h5' not in gChild: continue
            if 'dens' not in gChild and 'flux' not in gChild: continue
            #print(sChild,gChild)
            #continue
            for hlr in hlL:
                if hlr not in gChild: continue
                tmpD[hlr].append(gChild)
        genD[cubeN]={'path':sChildPath}
        genD[cubeN]['HR']=[tmpD['HR'][0],tmpD['HR'][-1]]
        genD[cubeN]['LR']=[tmpD['LR'][-1]]
        
        #break            
    return genD

#...!...!..................
def invert_tripack(triD):
    #print(list(triD))
    # computes flux from tau_red 
    nameD={
           'tau_red_HR_z3': 'flux_HR_z3',
           'tau_red_LR_z3': 'flux_LR_z3'}
    assert len(triD)==2 ,' expects no more cubes'
    for x in nameD:
        cube=triD.pop(x)
        shs=cube.shape
        print('inv %s shape %s'%(x,shs),cube.dtype)
        for i in range(shs[0]):
            cube[i]=np.exp(-cube[i])
        print('min/max:',np.min(cube),np.max(cube),nameD[x])
        triD[nameD[x]]=cube

 
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    hlL=['HR','LR']
    recMap={
            'HR':('derived_fields','tau_red') ,
            'LR':('derived_fields','tau_red') ,
    }


    genD={}
    genD['L80_s1']={'LR':'L80_N512_z3_s1.hdf5','HR':'L80_N4096_z3_s1.hdf5'}
    genD['L80_s2']={'LR':'L80_N512_z3_s2.hdf5','HR':'L80_N4096_z3_s2.hdf5'}
    #genD['L80_s1']={'LR':'L80_N512_z3_s1.hdf5','HR':'L80_N512_z3_s2.hdf5'}  # tmp-test
    ipath='/pscratch/sd/z/zarija/MLHydro/'

    pprint(genD) 

       
    nTri=0
    for cubeN in genD:  # loop over H5-files
        verb=1
        print('M: assemble ',cubeN)
        
        bigD={}
        for hlr in recMap:
            fileN=genD[cubeN][hlr]
            print('fileN',fileN)
            groupN,fieldN=recMap[hlr]
            inpF=os.path.join( ipath,fileN)
            big,meta=read_one_nyx_h5(inpF,[fieldN],groupN=groupN,verb=verb); verb=0
            pprint(meta)
            key=hlr+'m'
            if key not in genD[cubeN]: genD[cubeN][key]=meta
           
            zRed=meta['redshift']
            for x in big.keys():
                name='%s_%s_z%.0f'%(x,hlr,zRed)
                print('M: add',name)
                bigD[name]=big[x]
        # construct common  meta-data

        print('M:bigD=',sorted(bigD))
        invert_tripack(bigD) # compute fluxes from red_tau
 
        cs={ hlr:genD[cubeN][hlr+'m']['cell_size']  for hlr in hlL }
        bs={ hlr:genD[cubeN][hlr+'m']['cube_shape'][0]  for hlr in hlL }
     
        
        seed=cubeN.split('_')[1]
        #print(rs,seed)
        #pprint(cs)
        #pprint(bs)
        meta.pop('cube_shape')
        metaD=meta
        metaD['redshift']=int('%.0f'%genD[cubeN][hlr+'m']['redshift'])
        metaD['cell_size']=cs
        metaD['cube_size']=float( '%.1f'%metaD['cube_size'])
        metaD['cube_bin']=bs
        metaD['ic_seed']=seed
        pprint(metaD)
        outF=os.path.join(args.outPath,'flux_'+cubeN+'.twopack.h5')
        write4_data_hdf5(bigD,outF, metaD=metaD)
        nTri+=1
    print('M: done, nTri=',nTri)
        
