#!/usr/bin/env python3
'''
Verify, one cube 

'''

import scipy.stats as stats
from inspect_NyxHydroH5 import read_one_nyx_h5
import numpy as np
import argparse,os,time
from pprint import pprint

#...!...!..................
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v","--verbosity",type=int,choices=[0, 1, 2, 3], help="increase output verbosity", default=1, dest='verb')

    parser.add_argument( "-i","--index", default=33,type=int,help="image index")

    parser.add_argument( "-X","--noXterm", action='store_true', default=False, help="disable X-term for batch mode")
    parser.add_argument("--outPath", default='out/',help="output path for plots and tables")

    args = parser.parse_args()
    
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

def power_2Dcloud(image):
    npix = image.shape[0]
    print('M: img',image.shape)

    fourier_image = np.fft.fftn(image)
    fourier_amplitudes = np.abs(fourier_image)**2

    kfreq = np.fft.fftfreq(npix) * npix
    kfreq2D = np.meshgrid(kfreq, kfreq)
    knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

    knrm = knrm.flatten()
    fourier_amplitudes = fourier_amplitudes.flatten()

    kbins = np.arange(0.5, npix//2+1, 1.)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,  statistic = "mean", bins = kbins)
    Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)
    return kvals,Abins  # k,P(k)

     
#=================================
#=================================
#  M A I N 
#=================================
#=================================

if __name__ == "__main__":
    args=get_parser()
    plt=mini_plotter(args)

    dataPath='/pscratch/sd/b/balewski/tmp_NyxProd/2760607_2univ/cube_82337823'
    
    #.......... define Nyx input data
    inpF="plotLR00354_converted.h5"   # LR z=5
    inpF="plotLR00396_converted.h5"   # LR z=3
    
    inpF="plotHR00402_converted.h5"   # HR z=5
    #
    inpF="plotHR00001_converted.h5"   # HR z=3
    inpF="plotHR00475_converted.h5"   # HR z=3
    inpF="plotHR00704_converted.h5"   # HR z=3
    print('M: inpF',inpF)

    fieldN="baryon_density"
    bigD,meta=read_one_nyx_h5(os.path.join(dataPath,inpF),fieldN)
          
    pprint(meta)
    data=bigD[fieldN][args.index]
    print('img min/max',np.min(data), np.max(data))
    print(' mean, std',np.mean(data), np.std(data))
        
    print('hr0:',data.shape)


    # FFT
    k,P=power_2Dcloud(data)

    plDD={}
    plDD[1]=[inpF,np.log(1+data)]
    #plDD[1]=['rnd crop %d'%image_size,np.log(1+data1) ]
    
    # - - - - just plotting - - - -

    ncol,nrow=2,1; figId=4
    plt.figure(figId,facecolor='white', figsize=(6.5,3))

    # density
    ax=plt.subplot(nrow,ncol,1)
    binsX=np.linspace(0.,12,50)
    binsX=50
    t0=time.time()
    ax.hist(plDD[1][1].flatten(),binsX)#,density=True)
    ax.set_yscale('log')
    print(' histo done,   elaT=%.1f sec'%((time.time() - t0)))
    ax.grid()
    tit='slice %d'%(args.index)
    ax.set(title=tit,xlabel='log(1+rho)',ylabel='num bins')

    # ... fft
    ax=plt.subplot(nrow,ncol,2)
    #ax.loglog(k, P)
    ax.plot(k, P)
    ax.set_yscale('log')
    ax.set(xlabel="k",ylabel='P(k)',title='power spectrum')
   
    ax.grid()
    ax.set(xlabel='k', ylabel='P(k)',title='FFT strength')

    save_fig(figId)
    #plt.show()
    
    vmax=1. # cut-off for log(intensity)
    ncol,nrow=2,1; figId=5
    plt.figure(figId,facecolor='white', figsize=(13,6))
    for i in range(ncol*nrow):    
        ax=plt.subplot(nrow,ncol,1+i)
        ax.set_aspect(1.)
        txt,logD=plDD[1]
        print('log min/max',np.min(logD), np.max(logD))
        print(txt,logD.shape)
        ax.imshow(logD.T,vmax=vmax)
        tit='zRed=%.1f slice %d,  %s'%(meta['redshift'],args.index,txt)
        ax.set(title=tit)
        #if i==0: ax.grid()
        break
    save_fig(figId)

    plt.show()
