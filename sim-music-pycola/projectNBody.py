#!/usr/bin/env python3
__author__ = "Jan Balewski"
__email__ = "janstar1122@gmail.com"

# project DM vector field to density
import os,sys,time
import configparser
import numpy as np
import h5py    
from pprint import pprint
from toolbox.Util_H5io3 import read3_data_hdf5, write3_data_hdf5

import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2],
                        help="increase output verbosity", default=1, dest='verb')
    parser.add_argument("--inpPath",default='out')
    parser.add_argument("--outPath", default='same')
    parser.add_argument("--simConf",  default='univ_base0', help=" [.ics.conf] music config name")
    parser.add_argument("--level",default=7, type=int, help=" chose resolution from Music input")
        
    parser.add_argument("--extName",  default='music',choices=['music','pycola'], help="data name extesion ")
    args = parser.parse_args()

    if args.outPath=='same': args.outPath=args.inpPath
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    return args

#...!...!..................
def projectOne(data,level,boxlength):
    nbins=1<<level
   
    px = data['px']
    py = data['py']
    pz = data['pz']

    print ('pxyz: ',px[0][0][0], py[0][0][0], pz[0][0][0])
   
    #### Try using this hp.histogramdd function...
    ### For this I need to turn the particl elists into coord lists, 
    ### so (  (px[i][j][k], py[i][j][k], pz[i][j][k]), ....)
    pxf = np.ndarray.flatten(px)
    pyf = np.ndarray.flatten(py)
    pzf = np.ndarray.flatten(pz)

    print ('pxf.shape', pxf.shape)
    print ('pxf sample', pxf[0], pyf[0], pzf[0])
    print ('pxf min/max', pxf.min(), pxf.max())
    print ('pyf min/max', pyf.min(), pyf.max())
    print ('pzf min/max', pzf.min(), pzf.max())

    ### so the flattening is working. Now make this into a 3d array...
    ps = np.vstack( (pxf, pyf, pzf) ).T
    
    del(pxf); del(pyf); del(pzf)

    print ("one big vector list ", ps.shape, ps[77,:],'\naccumulate 3D histo...')
    
    ## OK! Then this is indeed a big old array. Now I want to histogram it.
    ## this step goes from a set of parcile coordinates to a histogram of particle counts 
    
    H, bins = np.histogramdd(ps, nbins, range=((0,boxlength),(0,boxlength),(0,boxlength)) )
 
    print ("histo shape", H.shape, 'example:', H[5][5][5])
    print ('mass sum=%.3g, min=%.3g max=%.3g'%(np.sum(H),H.min(),H.max())) # takes many seconds, just for QA
    
    return H.astype('float32')

#...!...!..................
def testOne(data):
    px = data['px']
    py = data['py']
    pz = data['pz']
 
    print ('ttt pxyz: ',px[0][0][0], py[0][0][0], pz[0][0][0])
   
    pxf = np.ndarray.flatten(px)
    pyf = np.ndarray.flatten(py)
    pzf = np.ndarray.flatten(pz)

    nd=pxf.shape[0]
    k=0
    for i in range(nd):
        #if pxf[i]<0.1: continue
        print(i,"%.2f, %.2f, %.2f"%(pxf[i],pyf[i],pzf[i])) 
        k+=1
    print('see %d non-zero of %d'%(k,nd))
    
#...!...!..................
def rdMusicH5(inpF,level,boxsize):
    # match import_music_snapshot(.), /usr/local/lib/python3.8/dist-packages/pycola3/ic.py
    core='level_0%02d_DM_d'%level
    print('rdMusicH5:',inpF, 'core:',core)
    # fill lists creating numpy arrays
    h5f = h5py.File(inpF, 'r') # read file
    #aa=h5f['level_000_DM_dx'][:]
    data={}
    data['px']=h5f[core+'x'][4:-4, 4:-4, 4:-4] * boxsize
    data['py']=h5f[core+'y'][4:-4, 4:-4, 4:-4] * boxsize
    data['pz']=h5f[core+'z'][4:-4, 4:-4, 4:-4] * boxsize
    h5f.close()
    aa=data['px']
    print('px shape',aa.shape,aa.dtype)
    k=3
    for i in range(k):
        j=i+6
        print(i,data['px'][j,j,j],data['py'][j,j,j],data['pz'][j,j,j])
    return data

#...!...!..................
def rdColaH5(inpF):
    print('rdColaH5:',inpF)
    # fill lists creating numpy arrays
    h5f = h5py.File(inpF, 'r') # read file
    #aa=h5f['level_000_DM_dx'][:]
    data={}
    data['px']=h5f['px'][:]
    data['py']=h5f['py'][:]
    data['pz']=h5f['pz'][:]
    h5f.close()
    aa=data['px']
    print('px shape',aa.shape,aa.dtype)
    k=3
    for i in range(k):
        j=i+6
        print(i,data['px'][j,j,j],data['py'][j,j,j],data['pz'][j,j,j])
    return data

#...!...!..................
def cfg2dict(cfg):
    metaD={}
    #print('sections:',cfg.sections())
    for sectN in cfg.sections():
        sectObj=cfg[sectN]
        sect={}
        metaD[sectN]=sect
        #print('\nsubs:',sectObj)
        for k,v in sectObj.items():
            #print(k,v)
            sect[k]=v

    return metaD
#=================================
#=================================
#  M A I N 
#=================================
#=================================
if __name__=="__main__":

    args=get_parser()
    cfgF=os.path.join(args.inpPath,args.simConf+'.ics.conf')
    cfg = configparser.RawConfigParser()   
    cfg.read(cfgF)
    print('conf sections:',cfg.sections(),cfgF)
    assert len(cfg.sections())>0
    boxlength = int(cfg['setup']['boxlength'])  # in Mpc/h
    musF = cfg['output']['filename']
    metaD=cfg2dict(cfg)
    
    if 'music' in args.extName:
        inpF=os.path.join(args.inpPath,musF)
        levMin=int(cfg['setup']['levelmin'])
        levMax=int(cfg['setup']['levelmax'])
        assert args.level>=levMin
        assert args.level<=levMax
        data=rdMusicH5(inpF,args.level,boxlength)
        metaD.pop('pycola')
        outF=musF.replace('.h5','.dm.h5')

    if 'pycola' in args.extName:
        colaF=musF.replace('.music','.pycola%d'%args.level)
        inpF=os.path.join(args.inpPath,colaF)
        data=rdColaH5(inpF)
        metaD['pycola']['filename']=colaF
        outF=colaF.replace('.h5','.dm.h5')
        
    #testOne(data)
    rho=projectOne(data,args.level,boxlength)
    bigD={'music':rho}
    
    outF=os.path.join(args.outPath,outF)
    write3_data_hdf5(bigD,outF,metaD=metaD)
    #pprint(metaD)
    print('M: done')
