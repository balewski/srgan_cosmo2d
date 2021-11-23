#!/usr/bin/env python3
'''
  compare  power spectrum  HR vs. SR

'''
import sys,os
from toolbox.Util_H5io3 import  read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml
from toolbox.Util_Cosmo2d import  powerSpect_2Dfield_numpy

import numpy as np
import argparse,os
import scipy.stats as stats
from pprint import pprint

from  scipy import signal
from calib_power import median_conf_V

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-e","--expName",default=None,help="(optional), append experiment dir to data path")
    parser.add_argument("-s","--genSol",default="last",help="generator solution")
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("-d","--dataPath",  default='/global/homes/b/balewski/prje/tmp_NyxHydro4kD/',help='data location w/o expName')
 
    args = parser.parse_args()
    if args.expName!=None:
        args.dataPath=os.path.join(args.dataPath,args.expName)
    
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args

#...!...!..................
def mini_plotter(args):
    import matplotlib as mpl
    if args.noXterm:
        mpl.use('Agg')  # to plot w/o X-server
        print('Graphics disabled')
    else:
        mpl.use('TkAgg') 
        print('Graphics started, canvas will pop-out')
    import matplotlib.pyplot as plt
    return plt

#...!...!..................
def save_fig( fid, ext='my', png=1):
     plt.figure(fid)
     plt.tight_layout()
     
     figName=args.outPath+'%s_f%d'%(ext,fid)
     if png: figName+='.png'
     else: figName+='.pdf'
     print('Graphics saving to ',figName)
     plt.savefig(figName)
     
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    plt=mini_plotter(args)
    
    #.......... input data
    inpF=os.path.join(args.dataPath,'pred-test-%s.h5'%args.genSol)
    fieldD,expMD=read3_data_hdf5(inpF)
    print('expMD:'); pprint(expMD)

    # assembly meta data

    #.... recover  data
    
    HR=fieldD['hr'][:,0]  # skip C-index
    SR=fieldD['sr'][:,0]
    space_step=expMD['field2d']['hr']['space_step']  # the same for SR
    nSamp=HR.shape[0]
    #nSamp=41
    Y=[]
    for i in range(nSamp):
        kphys,kidx,Phr,fftA2=powerSpect_2Dfield_numpy(HR[i],d=space_step)
        _,_,Psr,_=powerSpect_2Dfield_numpy(SR[i],d=space_step)
        
        Pres=(Psr-Phr)/Phr
        Y.append(Pres)
    
    Y=np.array(Y)
    Ymed=median_conf_V(Y)
    Yavr=np.mean(Y,axis=0)
    Ystd=np.std(Y,axis=0)
    
    print('M:Ymed',Ymed.shape,Yavr.shape)
    # smooth it
    for i in range(3):
        Ymed[i]=signal.savgol_filter(Ymed[i], window_length=11, polyorder=2, deriv=0)
        Yavr=signal.savgol_filter(Yavr, window_length=11, polyorder=2, deriv=0)
        Ystd=signal.savgol_filter(Ystd, window_length=11, polyorder=2, deriv=0)        
    
    # - - - - - Plotting - - - - - 
    plDD={}
    ncol,nrow=2,1; figId=6
    #tagN='%s-%s'%(args.expName,args.genSol)
    tagN=args.genSol
    
    
    if 1:
        plt.figure(figId,facecolor='white', figsize=(12,6))
        # - - - -  power 
        ax=plt.subplot(nrow,ncol,1)
        
        for i in range(nSamp):
            ax.step(kidx,Y[i], linewidth=1. )
        tit='%s,  relative P(k) residua,    nSamp=%d'%(tagN,nSamp)

        if 1:
            ax.plot(kidx,Ymed[0],linewidth=3,color='k',linestyle='--',label='median')
            ax.plot(kidx,Ymed[1],linewidth=3,color='k',linestyle=':')
            ax.plot(kidx,Ymed[2],linewidth=3,color='k',linestyle=':',label='med+/-std')
            
        if 1:
            ax.plot(kidx,Yavr,linewidth=3,color='gold',linestyle='--',label='average')
            ax.plot(kidx,Yavr-Ystd,linewidth=3,color='gold',linestyle=':')
            ax.plot(kidx,Yavr+Ystd,linewidth=3,color='gold',linestyle=':',label='avr+/-std')
            

        ax.legend(loc='best', title='summary stats')
            
        ax.set(title=tit, xlabel='wavenumber index',ylabel=' (P(k)SR - P(k)HR) / P(k)HR')
        ax.axhline(0,linestyle='--')
        ax.grid()
        ax.set_ylim(-1.,1.)

        save_fig(figId,ext=tagN)
    
    plt.show()
