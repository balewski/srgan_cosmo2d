#!/usr/bin/env python3
'''
  compare  power spectrum  HR vs. SR

'''
import sys,os
from toolbox.Util_H5io3 import  read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml
from toolbox.Util_Cosmo2d import  density_2Dfield_numpy,powerSpect_2Dfield_numpy, median_conf_V, median_conf_1D, srgan2d_FOM1

import numpy as np
import argparse,os
import scipy.stats as stats
from pprint import pprint

from  scipy import signal


PLOT={'density':1, 'fft':1, 'integral':1}

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-e","--expName",default=None,help="(optional), append experiment dir to data path")
    parser.add_argument("-s","--genSol",default="last",help="generator solution")
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("-d","--dataPath",
                        #default='/global/homes/b/balewski/prje/tmp_srganA/'
                        default='/pscratch/sd/b/balewski/tmp_NyxHydro512A/'
                        ,help='data location w/o expName')
 
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
    print('M:Ymed',Ymed.shape,Yavr.shape,'Y:',Y.shape)
    return Ymed,Yavr,Ystd  # skip filtering, tmp
    
    for i in range(3): # smooth it
        ##1print('Ymed-',i,Ymed)
        Ymed[i]=signal.savgol_filter(Ymed[i], window_length=11, polyorder=2, deriv=0)
    Yavr=signal.savgol_filter(Yavr, window_length=11, polyorder=2, deriv=0)
    Ystd=signal.savgol_filter(Ystd, window_length=11, polyorder=2, deriv=0)     
    return Ymed,Yavr,Ystd


#...!...!..................
def plot_stats(ax,X,Y,Ymed,Yavr=None,Ystd=None):

    N=min(40,nSamp)
    for i in range(N):
        # smooth it
        Ys=signal.savgol_filter(Y[i], window_length=5, polyorder=1, deriv=0)
        ax.step(X,Ys, linewidth=1. ) # individual distributions

    if 1:
        ax.plot(X,Ymed[0],linewidth=3,color='k',linestyle='--',label='median')
        ax.plot(X,Ymed[1],linewidth=3,color='k',linestyle=':')
        ax.plot(X,Ymed[2],linewidth=3,color='k',linestyle=':',label='med+/-std')

    if isinstance(Yavr, np.ndarray):
        ax.plot(X,Yavr,linewidth=3,color='gold',linestyle='--',label='average')
        ax.plot(X,Yavr-Ystd,linewidth=3,color='gold',linestyle=':')
        ax.plot(X,Yavr+Ystd,linewidth=3,color='gold',linestyle=':',label='avr+/-std')


    #ax.legend(loc='best', title='summary stats')
    ax.legend(loc='upper left')
    ax.axhline(1,linestyle='--')
    ax.grid()
    ax.set_ylim(0.4,1.6)
    #ax.set_ylim(0.1,1.9)

#...!...!..................
def plot_integrals(ax,HR,SR,tit):
    sum1=HR.shape[1]**2 
    assert str(HR.shape)==str(SR.shape)
    msum_hr=np.sum(HR,axis=(1,2))-sum1
    #print('ss',msum_hr.shape,msum_hr)
    msum_sr=np.sum(SR,axis=(1,2))-sum1
    #print('ss2',msum_sr.shape,msum_sr,sum1)
    rsum=msum_sr/msum_hr
    #print('ss3',rsum)

    # scale mass
    msum_hr/=1e6
    msum_sr/=1e6
    
    ncol,nrow=1,2; 
    ax=plt.subplot(nrow,ncol,1)
    binsX=50
    ax.hist(msum_sr, bins=binsX,label='SR',color='r') 
    ax.hist(msum_hr, histtype='step', bins=binsX,label='HR',color='k') 

    ax.grid()
    ax.set(title=tit, ylabel='images', xlabel='integral (mass*1e6)')
    ax.legend(loc='best')

    ax=plt.subplot(nrow,ncol,2)
    binsX=np.linspace(0.97,1.03,20)
    ax.hist(rsum, bins=binsX)
    ax.grid()
    ax.set( ylabel='images', xlabel='SR/HR integral mass')
    med,mstd,pstd=median_conf_1D(rsum)
    
    txt='median %.3f \nstd [ %.3f, %.3f ]'%(med,mstd,pstd)
    print('txt',txt)
    ax.text(0.1,0.8,txt,transform=ax.transAxes,color='b')
    ax.axvline(med,linewidth=1., color='k')
    ax.axvline(med+mstd,linewidth=1., linestyle='--', color='k')
    ax.axvline(med+pstd,linewidth=1., linestyle='--', color='k')

    
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
    fieldD,predMD=read3_data_hdf5(inpF)
    # filedD contains: rho+1
    #print('expMD:'); pprint(predMD)

    # assembly meta data

    #.... recover  data
    
    HR=fieldD['hrFin'][:,0]  # skip C-index, for now it is 1 channel
    SR=fieldD['srFin'][:,0]
    
    #space_step=eMD['field2d']['hr']['space_step']  # the same for SR
    space_step=predMD['inpMD']['cell_size']['HR']  # the same for SR
    nSamp=HR.shape[0]

    R=[] # rho-space
    P=[] # power spectrum space
    for i in range(nSamp):
        # ... compute density
        rphys,Rhr=density_2Dfield_numpy(np.log(HR[i]))
        _,Rsr=density_2Dfield_numpy(np.log(SR[i]))
        
        #print('Rsr-',i,Rhr.shape)
        r_rel=Rsr/Rhr
        R.append(r_rel)

        # ... compute power spectra
        kphys,kidx,Phr,fftA2=powerSpect_2Dfield_numpy(HR[i],d=space_step)
        _,_,Psr,_=powerSpect_2Dfield_numpy(SR[i],d=space_step)

        #print('Psr-',i,Phr.shape)
        p_rel=Psr/Phr
        P.append(p_rel)
        
    Rmed,Ravr,Rstd=do_stats(R)  # real space
    Pmed,Pavr,Pstd=do_stats(P)  # fourier space
    print('M:computed, plotting ...')

    # experiment w/ FOM
    fomD=srgan2d_FOM1(Rmed[0],Pmed[0])
    fomTxt='FOM:%.2g  = space:%.2g + fft:%.2g'%(fomD['fom'],fomD['r_fom'],fomD['f_fom'])
    print('M fom1:',fomTxt)
    # - - - - - Plotting - - - - -
    plDD={}

    tagN='%s-%s'%(args.expName,args.genSol)
    #tagN=args.genSol
    
    if PLOT['density']:  # - - - -  density
        ncol,nrow=2,1; figId=6
        plt.figure(figId,facecolor='white', figsize=(13,6))    
        ax=plt.subplot(nrow,ncol,1)
        plot_stats(ax,rphys,R,Rmed,Ravr,Rstd)
        tit='%s,  relative Density ,    nSamp=%d'%(tagN,nSamp)
        ax.set(title=tit, xlabel='ln(rho+1)',ylabel=' D(k)SR / D(k)HR' )
        ax.text(0.01,0.02,fomTxt,transform=ax.transAxes,color='k')
    
        if PLOT['fft']:  # - - - -  power 
            ax=plt.subplot(nrow,ncol,2)
            plot_stats(ax,kidx,P,Pmed,Pavr,Pstd)
            tit='%s,  relative Power Spectrum'%(tagN)
            ax.set(title=tit, xlabel='wavenumber index',ylabel=' P(k)SR / P(k)HR' )
            txt2='design='+predMD['modelDesign']
            ax.text(0.01,0.02,fomTxt,transform=ax.transAxes,color='k')

        save_fig(figId,ext=tagN)

    if PLOT['integral']:  # - - - -
        figId=7
        plt.figure(figId,facecolor='white', figsize=(4,6))
        tit='%s, nSamp=%d'%(tagN,nSamp)        
        plot_integrals(plt,HR,SR,tit)
        save_fig(figId,ext=tagN)
    
    plt.show()
