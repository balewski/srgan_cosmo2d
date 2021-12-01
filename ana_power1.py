#!/usr/bin/env python3
'''
  compare  power spectrum  HR vs. SR

'''
import sys,os
from toolbox.Util_H5io3 import  read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml
from toolbox.Util_Cosmo2d import  powerSpect_2Dfield_numpy,density_2Dfield_numpy

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
    parser.add_argument("-d","--dataPath",  default='/global/homes/b/balewski/prje/tmp_NyxHydro4kE/',help='data location w/o expName')
 
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
     

#...!...!..................
def do_stats(Y):
    Y=np.array(Y)
    Ymed=median_conf_V(Y)
    Yavr=np.mean(Y,axis=0)
    Ystd=np.std(Y,axis=0)
    print('M:Ymed',Ymed.shape,Yavr.shape)
    # smooth it
    for i in range(3):
        ##1print('Ymed-',i,Ymed)
        Ymed[i]=signal.savgol_filter(Ymed[i], window_length=11, polyorder=2, deriv=0)
    Yavr=signal.savgol_filter(Yavr, window_length=11, polyorder=2, deriv=0)
    Ystd=signal.savgol_filter(Ystd, window_length=11, polyorder=2, deriv=0)     
    return Ymed,Yavr,Ystd


#...!...!..................
def plot_stats(ax,X,Y,Ymed,Yavr=None,Ystd=None):

    N=min(30,nSamp)
    for i in range(N):
        ax.step(X,Y[i], linewidth=1. ) # individual distributions

    if 1:
        ax.plot(X,Ymed[0],linewidth=3,color='k',linestyle='--',label='median')
        ax.plot(X,Ymed[1],linewidth=3,color='k',linestyle=':')
        ax.plot(X,Ymed[2],linewidth=3,color='k',linestyle=':',label='med+/-std')

    if isinstance(Yavr, np.ndarray):
        ax.plot(X,Yavr,linewidth=3,color='gold',linestyle='--',label='average')
        ax.plot(X,Yavr-Ystd,linewidth=3,color='gold',linestyle=':')
        ax.plot(X,Yavr+Ystd,linewidth=3,color='gold',linestyle=':',label='avr+/-std')


    ax.legend(loc='best', title='summary stats')
    ax.axhline(1,linestyle='--')
    ax.grid()
    ax.set_ylim(0.4,1.6)

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
    # filedD contains: rho+1
    print('expMD:'); pprint(expMD)

    # assembly meta data

    #.... recover  data
    
    HR=fieldD['hr'][:,0]  # skip C-index
    SR=fieldD['sr'][:,0]
    
    space_step=expMD['field2d']['hr']['space_step']  # the same for SR
    nSamp=HR.shape[0]
    #nSamp=41
    R=[];P=[]
    for i in range(nSamp):
        # ... compute density
        rphys,Rhr=density_2Dfield_numpy(np.log(HR[i]))
        _,Rsr=density_2Dfield_numpy(np.log(SR[i]))
        
        #print('Rsr-',i,Rhr)
        Rrel=Rsr/Rhr
        R.append(Rrel)

        # ... compute power spectra
        kphys,kidx,Phr,fftA2=powerSpect_2Dfield_numpy(HR[i],d=space_step)
        _,_,Psr,_=powerSpect_2Dfield_numpy(SR[i],d=space_step)
        
        Prel=Psr/Phr
        P.append(Prel)
        
    Rmed,Ravr,Rstd=do_stats(R)
    Pmed,Pavr,Pstd=do_stats(P)
    print('M:computed, plotting ...')
   
    
    # - - - - - Plotting - - - - - 
    plDD={}
    ncol,nrow=2,1; figId=6
    tagN='%s-%s'%(args.expName,args.genSol)
    #tagN=args.genSol
    
    plt.figure(figId,facecolor='white', figsize=(12,6))
    if 1:  # - - - -  density
        ax=plt.subplot(nrow,ncol,1)
        plot_stats(ax,rphys,R,Rmed,Ravr,Rstd)
        tit='%s,  relative Density ,    nSamp=%d'%(tagN,nSamp)
        ax.set(title=tit, xlabel='ln(rho+1)',ylabel=' D(k)SR / D(k)HR' )
        #print('xxx',rphys)
        #print('Rmed',Rmed)
        #print('Ravr',Ravr)
    
    if 1:  # - - - -  power 
        ax=plt.subplot(nrow,ncol,2)
        plot_stats(ax,kidx,P,Pmed,Pavr,Pstd)
        tit='%s,  relative Power Spectrum'%(tagN)
        ax.set(title=tit, xlabel='wavenumber index',ylabel=' P(k)SR / P(k)HR' )

    save_fig(figId,ext=tagN)
    plt.show()
