#!/usr/bin/env python3
'''
  compare  power spectrum  HR vs. SR

'''
import sys,os
from toolbox.Util_H5io3 import  read3_data_hdf5
from toolbox.Util_IOfunc import write_yaml
from toolbox.Util_Cosmo2d import  density_2Dfield_numpy,powerSpect_2DfieldBin0_numpy , median_conf_V , median_conf_1D, srgan2d_FOM1

import numpy as np
import argparse,os
import scipy.stats as stats
from pprint import pprint

from  scipy import signal
from toolbox.Plotter_Backbone import Plotter_Backbone

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("-e","--expName",default=None,help="(optional), append experiment dir to data path")
    parser.add_argument("-s","--genSol",default="last",help="generator solution")
    parser.add_argument("-o","--outPath", default='out/',help="output path for plots and tables")
    parser.add_argument("-p", "--showPlots",  default='abc', nargs='+',help="abc-string listing shown plots")

    parser.add_argument("-d","--dataPath",
                        #default='/global/homes/b/balewski/prje/tmp_srganA/'
                        default='/pscratch/sd/b/balewski/tmp_NyxHydro512A/'
                        ,help='data location w/o expName')
 
    args = parser.parse_args()
    args.showPlots=''.join(args.showPlots)
    if args.expName!=None:
        args.dataPath=os.path.join(args.dataPath,args.expName)
    args.prjName=args.expName
    for arg in vars(args):  print( 'myArg:',arg, getattr(args, arg))
    if not os.path.exists(args.outPath):
        os.makedirs(args.outPath);   print('M: created',args.outPath)
    return args

#............................
#............................
#............................
class Plotter(Plotter_Backbone):
    def __init__(self,args):
        Plotter_Backbone.__init__(self,args)
        self.args=args

#...!...!..................
    def integrals(self,HR,SR,plDD,figId=4):
        figId=self.smart_append(figId)
        ncol,nrow=1,2;
        fig=self.plt.figure(figId,facecolor='white', figsize=(4,6))
        
        assert str(HR.shape)==str(SR.shape)
        msum_hr=np.sum(HR,axis=(1,2))
        msum_sr=np.sum(SR,axis=(1,2))
        rsum=msum_sr/msum_hr

        # scale mass
        msum_hr/=1e6
        msum_sr/=1e6
        
        ax=self.plt.subplot(nrow,ncol,1)
        binsX=50
        ax.hist(msum_sr, bins=binsX,label='SR',color='r') 
        ax.hist(msum_hr, histtype='step', bins=binsX,label='HR',color='k') 

        ax.grid()
        ax.set(title=tit, ylabel='images', xlabel='integral flux/1e6')
        ax.legend(loc='best')

        ax=self.plt.subplot(nrow,ncol,2)
        binsX=np.linspace(0.97,1.03,20)
        ax.hist(rsum, bins=binsX)
        ax.grid()
        ax.set( ylabel='images', xlabel='SR/HR integral flux')
        med,mstd,pstd=median_conf_1D(rsum)

        txt='median %.3f \nstd [ %.3f, %.3f ]'%(med,mstd,pstd)
        print('txt',txt)
        ax.text(0.1,0.8,txt,transform=ax.transAxes,color='b')
        ax.axvline(med,linewidth=1., color='k')
        ax.axvline(med+mstd,linewidth=1., linestyle='--', color='k')
        ax.axvline(med+pstd,linewidth=1., linestyle='--', color='k')

#...!...!..................
    def traces(self,X,Y,Ymed,Yavr=None,Ystd=None,figId=6,obsN=None):
        figId=self.smart_append(figId)
        ncol,nrow=1,1
        fig=self.plt.figure(figId,facecolor='white', figsize=(6,6))
        ax=self.plt.subplot(nrow,ncol,1)
        
        N=min(10,nSamp)
        for i in range(N):
            # smooth it
            Ys=signal.savgol_filter(Y[i], window_length=5, polyorder=1, deriv=0)
            ax.step(X,Ys, linewidth=1. ) # individual distributions

        if 1:
            ax.plot(X,Ymed[0],linewidth=3,color='k',linestyle='--',label='median')
            ax.plot(X,Ymed[1],linewidth=3,color='k',linestyle=':')
            ax.plot(X,Ymed[2],linewidth=3,color='k',linestyle=':',label='med+/-std')

        if isinstance(Yavr, np.ndarray) and 0:
            ax.plot(X,Yavr,linewidth=3,color='gold',linestyle='--',label='average')
            ax.plot(X,Yavr-Ystd,linewidth=3,color='gold',linestyle=':')
            ax.plot(X,Yavr+Ystd,linewidth=3,color='gold',linestyle=':',label='avr+/-std')


        #ax.legend(loc='best', title='summary stats')
        ax.legend(loc='upper left')
        ax.axhline(1,linestyle='--')
        ax.grid()
        tit='%s,  relative %s ,    nSamp=%d'%(tagN,obsN,nSamp)
        if obsN=='flux':
            ax.set_ylim(0.4,1.6)
            ax.set(title=tit, xlabel='flux/pixel',ylabel=' D(flux)SR / D(flux)HR' )
        else:
            ax.set_ylim(0.1,10.)
            ax.set(title=tit, xlabel='k(z*)',ylabel=' power(SR) / power(HR)' )
        ax.text(0.01,0.02,fomTxt,transform=ax.transAxes,color='k')
    
     

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



    
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    
    #.......... input data
    inpF=os.path.join(args.dataPath,'pred-test-%s.h5'%args.genSol)
    fieldD,predMD=read3_data_hdf5(inpF)
    # filedD contains:flux
    #print('expMD:'); pprint(predMD)
    #.... recover  data    
    HR=fieldD['hrFin'][:,0]  # skip C-index, for now it is 1 channel
    SR=fieldD['srFin'][:,0]
    
    if 0:  #reduce num samples for debugging
        HR=HR[:100]; SR=SR[:100]
        
    space_step=predMD['inpMD']['cell_size']['HR']  # the same for SR
    nSamp=HR.shape[0]

    R=[] # flux-space
    P=[] # power spectrum space
    for i in range(nSamp):
        # ... compute density
        rphys,Rhr=density_2Dfield_numpy(HR[i])
        _,Rsr=density_2Dfield_numpy(SR[i])
        
        r_rel=Rsr/Rhr
        R.append(r_rel)

        # ... compute power spectra
        kphys,kidx,Phr=powerSpect_2DfieldBin0_numpy(HR[i],d=space_step)
        _,_,Psr    =powerSpect_2DfieldBin0_numpy(SR[i],d=space_step)
        #print('pp',Psr.shape,kphys[::5],kidx[::5]); aa1
        
        p_rel=Psr/Phr
        P.append(p_rel)
        
    Rmed,Ravr,Rstd=do_stats(R)  # flux space
    Pmed,Pavr,Pstd=do_stats(P)  # fourier space
    print('M:computed, plotting ...')

    # experiment w/ FOM
    fomD=srgan2d_FOM1(Rmed[0],Pmed[0])
    fomTxt='FOM:%.2g  = space:%.2g + fft:%.2g'%(fomD['fom'],fomD['r_fom'],fomD['f_fom'])
    print('M fom1:',fomTxt)
    # - - - - - Plotting - - - - -
    plDD={}

    tagN='%s-%s'%(args.expName,args.genSol)
    tit='%s, nSamp=%d'%(tagN,nSamp)
    
    # - - - - - PLOTTER - - - - -
    plot=Plotter(args)
    if 'a' in args.showPlots:     plot.integrals(HR,SR,tit)
    if 'b' in args.showPlots:     plot.traces(rphys,R,Rmed,Ravr,Rstd,obsN='flux')
    if 'c' in args.showPlots:     plot.traces(kidx,P,Pmed,Pavr,Pstd,obsN='power')

    plot.display_all('sr_sum')
  
